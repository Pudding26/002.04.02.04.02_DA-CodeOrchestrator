import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any

from contextlib import contextmanager


import numpy as np
import h5py
import pandas as pd


from multiprocessing import Process, Queue, Event, Manager
from multiprocessing import JoinableQueue


from app.utils.logger.loggingWrapper import LoggingHandler


from app.tasks.TaskBase import TaskBase
from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.dataModels.Jobs.JobEnums import JobStatus
from app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler
from app.tasks.TA41_ImageSegmentation.TA41_A_Segmenter import TA41_A_Segmenter
from app.utils.dataModels.Jobs.SegmenterJob import SegmenterJob
from app.utils.dataModels.Jobs.ExtractorJob import ExtractorJobInput

from app.utils.HDF5.HDF5_Inspector import HDF5Inspector

from app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out
from app.utils.SQL.models.production.api_SegmentationResults import SegmentationResults_Out

# ==================== GLOBAL FUNCTIONS ====================
LoggingHandler(logging_level="DEBUG-2")



def loader(
        worker_id: int,
        input_queue,
        output_queue,
        error_counter,
        error_lock,
        ready_to_read,
        jobs_len: int,
        error_threshold: int,
        setup_start_time
    ):
    from app.tasks.TA41_ImageSegmentation.TA41_B_FeatureProcessor import TA41_B_FeatureProcessor
    feature_processor = TA41_B_FeatureProcessor()
    logging.debug(f"[Loader-{worker_id}] Started")
    ellapsed_setup = time.time() - setup_start_time
    
    while True:
        job = input_queue.get()
        if job is None:
            input_queue.task_done()
            logging.debug(f"[Loader-{worker_id}] Exiting")
            break

        with error_lock:
            if error_counter['count'] >= error_threshold:
                input_queue.task_done()
                logging.warning(f"[Loader-{worker_id}] Error threshold exceeded")
                continue

        try:
            if job.input.job_No % 25 == 0 or job.input.job_No == 0 or job.input.job_No == jobs_len - 1:
                logging.debug2(f"[Loader-{worker_id}] Processing job #{job.input.job_No}/{jobs_len}")

            ready_to_read.wait()
            hdf5 = SWMR_HDF5Handler(job.input.hdf5_path)
            image_stack = hdf5.load_image(job.input.src_file_path)

            attrs_raw = job.attrs.attrs_raw.copy()
            attrs_raw.pop("parent_job_uuids", None)
            attrs_raw.update({"colorDepth": "8bit", "colorSpace": "GS", "filterNo": job.input.dest_FilterNo})
            job.attrs.attrs_FF = {**attrs_raw, "filterNo": "FF"}
            job.attrs.attrs_GS = {**attrs_raw, "filterNo": "GS"}

            segmentor = TA41_A_Segmenter(
                config=job.input.filter_instructions,
                image_stack=image_stack,
                image_stack_id=job.input.dest_stackID_FF,
                gpu_mode=False,
            )



     

            start_time = time.time()
            result = segmentor.run_stack()
            
            time_seg = time.time()
            feature_df = feature_processor.process_all(
                jobs= 
                    [
                    (job.input.dest_stackID_FF, result.get("features"))
                    ]
                )

            time_binning = time.time()
            


            job.input.image_FF = np.stack(result.get("filtered_image_stack"))
            job.input.image_GS = np.stack(result.get("new_gray_stack"))
            job.attrs.features_df = feature_df


            n, h, w = job.input.image_FF.shape

            job.stats = {
                "stackID_FF": job.input.dest_stackID_FF,
                "source": job.attrs.attrs_raw.get("sourceNo", "unknown"),
                "images": n,
                "height": h,
                "width": w,
                "filterNo": job.input.dest_FilterNo,
                "elapsed_seg": time_seg - start_time,
                "elapsed_binning": time_binning - time_seg,
                "elapsed_total": time.time() - start_time,
                "loader_id": f"Loader-{worker_id}",
                "ellapsed_setup": ellapsed_setup,
            }
            ellapsed_setup = 0

            output_queue.put(job)

        except Exception as e:
            logging.exception(f"[Loader-{worker_id}] Error: {e}")
            with error_lock:
                error_counter['count'] += 1
        finally:
            input_queue.task_done()


def storer(
    output_queue,
    error_counter,
    error_lock,
    hdf5_path,
    ready_to_read,
    stats_list: List[Dict[str, Any]] = []
):
    logging.debug2("[Storer] Started")
    handler = SWMR_HDF5Handler(hdf5_path)
    @contextmanager
    def suppress_logging(self, level=logging.WARNING):
        """
        Temporarily suppress logging below the given level (default: WARNING).

        Usage:
            with self.suppress_logging():
                self.fetch(...)
        """
        logger = logging.getLogger()
        previous_level = logger.level
        logger.setLevel(level)
        try:
            yield
        finally:
            logger.setLevel(previous_level)


    with h5py.File(handler.file_path, "a", libver="latest") as f:
        f.swmr_mode = True
        ready_to_read.set()

        last_job = -1
        while True:
            job = output_queue.get()
            if job is None:
                output_queue.task_done()
                logging.debug2(f"[Storer] Exiting after last job #{last_job}")
                break

            last_job = job.input.job_No
            try:
                start_time = time.time()
                woodMaster_updates = []
                handler.handle_dataset(
                    hdf5_file=f,
                    dataset_name=job.input.dest_file_path_FF,
                    numpy_array=job.input.image_FF,
                    attributes_new=job.attrs.attrs_FF,
                    attribute_process="att_replace",
                )
                woodMaster_updates.append(job.input.dest_file_path_FF)

                job.input.image_FF = None

                if job.input.image_GS and not isinstance(job.input.image_GS[0], type(None)):
                    handler.handle_dataset(
                        hdf5_file=f,
                        dataset_name=job.input.dest_file_path_GS,
                        numpy_array=job.input.image_GS,
                        attributes_new=job.attrs.attrs_GS,
                        attribute_process="att_replace",
                    )
                    woodMaster_updates.append(job.input.dest_file_path_GS)

                job.input.image_GS = None

                f.flush()

                if job.attrs.segmentation_mask_raw is not None:
                    mask = job.attrs.segmentation_mask_raw
                    n, h, w = mask.shape
                    job.attrs.extractorJobinput = ExtractorJobInput(
                        mask=mask,
                        n_images=n,
                        width=w,
                        height=h,
                        stackID=job.input.dest_stackID_FF,
                    )
                    job.attrs.segmentation_mask_raw = None
                
                
                store_time_image = time.time()
                
                with suppress_logging(logging.WARNING):
                    SegmentationResults_Out.store_dataframe(
                        df=job.attrs.features_df,
                        method="append")
                    job.attrs.features_df = None
                store_time_results = time.time()
                
                HDF5Inspector.update_woodMaster_row(
                    hdf5_path=handler.file_path,
                    dataset_path=woodMaster_updates
                )
                
                job.status = JobStatus.DONE
                job.updated = datetime.now(timezone.utc)

                job.stats.update({
                    "elapsed_store_image": store_time_image - start_time,
                    "elapsed_store_results": store_time_results - start_time,
                    "elapsed_total": job.stats.get("elapsed_total", 0) + (store_time_results - start_time),
                })
                

                job.update_db(fields_to_update=["status", "payload"])
                stats_list.append(job.stats.copy())
                if len(stats_list) % 25 == 0:
                    _print_summary_df(list(stats_list))

            except Exception as e:
                logging.exception(f"[Storer] Error on job #{job.input.job_No}: {e}")
                with error_lock:
                    error_counter['count'] += 1
            finally:
                output_queue.task_done()




class TA41_0_SegmentationOrchestrator(TaskBase):
    def setup(self):
        logging.debug2("🔧 Running setup...")
        self.jobs = []
        self.controller.update_message("Loading job metadata")

    def run(self):
        try:
            logging.debug2("🚀 Starting main run loop")
            self.controller.update_message("Building job pipeline")
            self.load_jobs_from_db()
            if len(self.jobs) == 0:
                logging.warning("⚠️ No jobs found to process, exiting")
                self.controller.finalize_success()
                return

            num_workers = round(max(1, min(len(self.jobs) // 60, 6)))
            num_workers = 6

            logging.info(f"🔧 Using {num_workers} worker processes for processing")
            self._run_pipeline(self.jobs, num_loader_workers=num_workers, max_queue_size=50, error_threshold=3)
            self.controller.update_message("Finalizing WoodMaster")
            self.controller.finalize_success()
            logging.info("✅ Task completed successfully")
        except Exception as e:
            self.controller.finalize_failure(str(e))
            logging.error(f"❌ Task failed: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()
            logging.debug2("🧹 Cleanup completed")

    def cleanup(self):
        logging.debug2("🧹 Running cleanup")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def load_jobs_from_db(self):
        logging.debug2("📥 Loading SegmenterJobs from database")

        filter_model = FilterModel.from_human_filter({"contains": {"job_type": "segmenter"}})
        df = WorkerJobs_Out.fetch(filter_model=filter_model)

        total_raw_jobs = len(df)
        self.controller.update_item_count(total_raw_jobs)
        logging.debug2(f"📊 {total_raw_jobs} segmenter job records loaded from DB")

        retry_ready = 0
        retry_delayed = 0
        total_parsed = 0

        for row in df.to_dict(orient="records"):
            try:
                job = SegmenterJob.model_validate(row["payload"])
                total_parsed += 1
                job.job_uuid = row["job_uuid"]
                if job.next_retry <= datetime.now(timezone.utc):
                    self.jobs.append(job)
                    retry_ready += 1
                else:
                    retry_delayed += 1
            except Exception as e:
                logging.error(f"❌ Failed to parse SegmenterJob: {e}", exc_info=True)

        for job_no, job in enumerate(self.jobs):
            job.input.job_No = job_no

        logging.info("📦 Job Loading Summary")
        logging.info(f"  • Total jobs fetched from DB:        {total_raw_jobs}")
        logging.info(f"  • Successfully parsed SegmenterJobs: {total_parsed}")
        logging.info(f"  • Jobs ready to run (retry OK):     {retry_ready}")
        logging.info(f"  • Skipped (next_retry in future):   {retry_delayed}")

    def _run_pipeline(
        self,
        jobs: List[SegmenterJob],
        num_loader_workers: int = 4,
        max_queue_size: int = 50,
        error_threshold: int = 3,
    ):
        manager = Manager()
        ready_to_read = Event()
        input_queue = JoinableQueue()
        output_queue = JoinableQueue(maxsize=max_queue_size)
        error_counter = manager.dict(count=0)
        error_lock = manager.Lock()
        stats_list = manager.list()

        pipeline_stats = {
            "total_jobs": len(jobs),
            "total_images": 0,
            "error_count": 0,
            "per_source": defaultdict(lambda: {"jobs": 0, "images": 0, "elapsed": 0.0}),
        }

        

        storer_proc = Process(
            target=storer,
            args=(output_queue, error_counter, error_lock, self.instructions["HDF5_file_path"], ready_to_read, stats_list),
            daemon=True,
        )
        storer_proc.start()
        self.record_pid(storer_proc.pid)


        setup_start_time = time.time()
        loaders = [
            Process(
                target=loader,
                args=(i, input_queue, output_queue, error_counter, error_lock, ready_to_read, len(jobs), error_threshold, setup_start_time),
                daemon=True,
            )
            for i in range(num_loader_workers)
        ]

        for l in loaders:
            l.start()
            self.record_pid(l.pid)

        # Feed jobs
        for job in jobs:
            input_queue.put(job)

        # Tell loaders to stop
        for _ in loaders:
            input_queue.put(None)

        # No more puts after this
        input_queue.close()          # optional — just to prevent further puts
        input_queue.join_thread()    # optional — join feeder thread

        # Wait for all loader processes
        for l in loaders:
            l.join()

        # Tell storer to stop
        output_queue.put(None)

        # Close output queue and wait for join
        output_queue.close()
        output_queue.join_thread()

        # Wait for storer process
        storer_proc.join()
        summary_df = _create_pipeline_summary(list(stats_list))
        logging.debug2("\n" + summary_df.to_string(index=False))


def _print_summary_df(stats_list: List[Dict]):
    if not stats_list:
        logging.debug2("No stats to print")
        return

    summary_df =_create_pipeline_summary(stats_list)
    
    # Print summary
    logging.debug2("\n" + summary_df.to_string(index=False))

def _create_pipeline_summary(stats_list: Dict) -> pd.DataFrame:

    df = pd.DataFrame(stats_list)
    num_workers = df['loader_id'].nunique()  # unique workers
    setup_ratio = df.groupby('loader_id')['ellapsed_setup'].max().sum() / num_workers

    summary_df = df.groupby('source', as_index=False).agg(
        jobs=('source', 'count'),
        images=('images', 'sum'),
        width_min=('width', 'min'),
        width_max=('width', 'max'),
        height_min=('height', 'min'),
        height_max=('height', 'max'),
        elapsed_total=('elapsed_total', 'sum'),
        elapsed_seg=('elapsed_seg', 'mean'),
        elapsed_binning=('elapsed_binning', 'mean'),
        elapsed_store_image=('elapsed_store_image', 'mean'),
        elapsed_store_results=('elapsed_store_results', 'mean'),
    )
    summary_df["total_overhead"] = setup_ratio

    # Round elapsed time columns
    time_cols = ['elapsed_total', 'elapsed_seg', 'elapsed_binning',
                 'elapsed_store_image', 'elapsed_store_results']
    for col in time_cols:
        summary_df[col] = summary_df[col].map(lambda x: round(x, 2))

    # Compute rates
    summary_df['images_per_s'] = (summary_df['images'] / summary_df['elapsed_total']).round(2)
    summary_df['stacks_per_s'] = (summary_df['jobs'] / summary_df['elapsed_total']).round(2)
    summary_df['overhead_per_image'] = (summary_df["total_overhead"] / summary_df['images']).round(2)

    return summary_df