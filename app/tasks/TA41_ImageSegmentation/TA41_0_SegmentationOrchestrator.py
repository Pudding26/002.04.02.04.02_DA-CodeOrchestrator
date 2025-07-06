import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any

from contextlib import contextmanager


import numpy as np
import h5py
import pandas as pd

from threading import Thread
from multiprocessing import Process, Queue, Event, Manager
from multiprocessing import JoinableQueue
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)



from queue import Queue
from app.utils.dataModels.Jobs.SWMR_StorerJob import SWMR_StorerJob
from app.tasks.TA02_HDF5Storer.shared_queue import store_queue


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
        setup_start_time,
        loop_no: int = 0,

    ):
    from app.tasks.TA41_ImageSegmentation.TA41_B_FeatureProcessor import TA41_B_FeatureProcessor
    feature_processor = TA41_B_FeatureProcessor()
    logging.debug2(f"[Loader-{worker_id}] Started")
    ellapsed_setup = time.time() - setup_start_time
    
    while True:
        job = input_queue.get()
        if job is None:
            input_queue.task_done()
            logging.debug(f"[Loader-{worker_id}] Exiting")
            break

        with error_lock:
            if error_counter.value >= error_threshold:
                input_queue.task_done()
                logging.warning(f"[Loader-{worker_id}] Error threshold exceeded")
                continue

        try:
            if job.input.job_No % 250 == 0 or job.input.job_No == 0 or job.input.job_No == jobs_len - 1:
                logging.debug2(f"[Loader-{worker_id}] Processing job #{job.input.job_No}/{jobs_len} in loopNo # {loop_no}")

            #ready_to_read.wait()
            hdf5 = SWMR_HDF5Handler(job.input.hdf5_path)
            image_stack = hdf5.load_image(job.input.src_file_path)

            attrs_raw = job.attrs.attrs_raw.copy()
            attrs_raw.pop("parent_job_uuids", None)
            attrs_raw.update({"colorDepth": "8bit", "colorSpace": "GS", "filterNo": job.input.dest_FilterNo})
            job.attrs.attrs_FF = {**attrs_raw, "filterNo": job.input.dest_FilterNo}
            job.attrs.attrs_GS = {**attrs_raw, "filterNo": "GS"}

            segmentor = TA41_A_Segmenter(
                config=job.input.filter_instructions,
                image_stack=image_stack,
                image_stack_id=job.input.dest_stackID_FF,
                gpu_mode=False,
            )



     

            start_time = time.time()
            result = segmentor.run_stack()
            seg_stats = result.get("stats", {})
            
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
            
            def aggregate_timings(timings):
                keys = timings[0].keys()
                return {f"avg_{k}": float(np.sum([t[k] for t in timings])) for k in keys}

            avg_stats = aggregate_timings(result.get("stats", []))


            job.stats = {
                "stackID_FF": job.input.dest_stackID_FF,
                "source": job.attrs.attrs_raw.get("sourceNo", "unknown"),
                "images": n,
                "height": h,
                "width": w,
                "filterNo": job.input.dest_FilterNo,
                "avg_segmentation_time": avg_stats.get("avg_segmentation_time", 0.0),
                "avg_preprocessing_time": avg_stats.get("avg_preprocessing_time", 0.0),
                "avg_feature_extraction_time": avg_stats.get("avg_feature_extraction_time", 0),                
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
                error_counter.value += 1

        finally:
            input_queue.task_done()



def storer(
    output_queue,
    error_counter,
    error_lock,
    hdf5_path,
    stats_list,
    controller,
    jobs_len,
    loop_no,
):
    """
    Consumer process that receives segmentation jobs from the output queue,
    stores images via the centralized HDF5 storer, updates woodMaster metadata,
    stores segmentation results and input payloads for downstream extractor tasks,
    and updates job state.

    Parameters:
        output_queue (Queue): Queue of processed segmentation jobs to be stored.
        error_counter (dict): Shared error counter (with a lock).
        error_lock (Lock): Lock for safely incrementing the error count.
        hdf5_path (str): Target HDF5 file for storage.
        ready_to_read (Event): Signal flag for extractor readiness.
        stats_list (list): Shared list to collect per-job timing/statistics.
        controller (TaskController): Used to update progress and status in UI.
        jobs_len (int): Total number of jobs in this loop.
        loop_no (int): Current pipeline loop number (for display).
    """

    while True:
        job = output_queue.get()
        if job is None:
            output_queue.task_done()
            break

        try:
            # Prepare woodMaster update paths
            woodMaster_updates = [job.input.dest_file_path_FF]
            if job.input.image_GS is not None:
                woodMaster_updates.append(job.input.dest_file_path_GS)

            # Prepare and submit FF write job
            result_queue_ff = Queue()
            store_queue.put(SWMR_StorerJob(
                dataset_path=job.input.dest_file_path_FF,
                image_data=job.input.image_FF,
                attributes=job.attrs.attrs_FF,
                attribute_process="att_replace",
                handler_method="store_image",
                result_queue=result_queue_ff
            ))

            success_ff, result_ff = result_queue_ff.get(timeout=30)
            if not success_ff:
                raise result_ff
            job.input.image_FF = None

            # Prepare and submit GS write job (if present)
            if job.input.image_GS is not None and len(job.input.image_GS) > 0 and job.input.image_GS[0] is not None:
                result_queue_gs = Queue()
                store_queue.put(SWMR_StorerJob(
                    dataset_path=job.input.dest_file_path_GS,
                    image_data=job.input.image_GS,
                    attributes=job.attrs.attrs_GS,
                    attribute_process="att_replace",
                    handler_method="store_image",
                    result_queue=result_queue_gs
                ))

                success_gs, result_gs = result_queue_gs.get(timeout=30)
                if not success_gs:
                    raise result_gs
                job.input.image_GS = None

            # Update HDF5 woodMaster metadata
            HDF5Inspector.update_woodMaster_paths(
                hdf5_path=hdf5_path,
                dataset_paths=woodMaster_updates
            )

            # Store segmentation result DataFrame
            SegmentationResults_Out.store_dataframe(
                df=job.attrs.features_df,
                method="append",
        
            )
            job.attrs.features_df = None

            
    

            # Update job status
            job.status = JobStatus.DONE
            job.updated = datetime.now(timezone.utc)
            job.update_db(fields_to_update=["status"])

            # Report progress every 10 jobs
            if job.input.job_No % 10 == 0:
                progress = ((job.input.job_No + 1) / jobs_len) * 100
                controller.update_progress(progress)
                controller.update_message(f"Loop {loop_no}: job {job.input.job_No}/{jobs_len}")

            # Report stats (per job timing)
            if job.stats and "timing" in job.stats:
                stats_list.append(job.stats["timing"])

        except Exception as e:
            # Log and register failure
            job.register_failure(str(e))
            job.update_db(fields_to_update=["status", "attempts", "next_retry"])

            with error_lock:
                error_counter.value += 1

            logging.exception(f"[Storer] Failed to store job #{job.input.job_No}")

        finally:
            output_queue.task_done()





class TA41_0_SegmentationOrchestrator(TaskBase):
    def setup(self):
        logging.debug2("🔧 Running setup...")
        self.jobs = []
        self.controller.update_message("Loading job metadata")

    def run(self):
        
        loop_no = 1
        backoff = 0
        max_sleep = 600
        base_sleep = self.instructions.get("sleep_time", 60)
        
        try:
            while True:
                if self.controller.should_stop():
                    logging.info("🛑 Task stopped by controller")
                    self.controller.finalize_success()
                    break

                loop_start = datetime.now(timezone.utc)
                logging.debug2(f"🔁 Starting Loop #{loop_no}")
                self.controller.update_message(f"Loop #{loop_no} — loading jobs")

                self.jobs = []
                self.load_jobs_from_db()
                total_jobs = len(self.jobs)

                self.controller.update_item_count(total_jobs)

                if total_jobs == 0:
                    sleep_time = min(base_sleep * (2 ** backoff), max_sleep)
                    logging.info(f"🕒 No jobs — sleeping {sleep_time}s (backoff={backoff})")
                    time.sleep(sleep_time)
                    backoff += 1
                    loop_no += 1
                    continue

                backoff = 0  # reset backoff
                self.controller.update_message("Running segmentation pipeline")
                summary_df = self._run_pipeline(self.jobs,
                                                num_loader_workers=round(max(1, min(len(self.jobs) // 60, 6))),
                                                loop_no=loop_no)
                
                logging.debug5(
                        f"""
                    📊 Segmentation Summary — Loop #{loop_no}
                    🔢 Jobs: {total_jobs}
                    ⏱️  Duration: {round((datetime.now(timezone.utc) - loop_start).total_seconds(), 2)}s
                    {summary_df.to_string(index=False)}
                    """
                    )
                self.controller.update_message(f"Loop #{loop_no} done")
                

                logging.info(f"✅ Loop #{loop_no} done — processed {total_jobs} jobs")
                loop_no += 1
                time.sleep(base_sleep)
                
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

        filter_model = FilterModel.from_human_filter({"contains": {"job_type": "segmenter",
                                                                   "status": "ready"}})
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

    def _run_pipeline(self, jobs: list[SegmenterJob], loop_no: int = 0, num_loader_workers: int = 4) -> pd.DataFrame:

        error_threshold = 5  # or however many you want



        input_queue = multiprocessing.JoinableQueue()
        output_queue = multiprocessing.JoinableQueue()
        error_counter = multiprocessing.Value("i", 0)
        error_lock = multiprocessing.Lock()
        stats_list = multiprocessing.Manager().list()

        # Feed jobs into the input queue
        for job in jobs:
            input_queue.put(job)

        # Add sentinel objects for each worker to signal shutdown
        for _ in range(num_loader_workers):
            input_queue.put(None)

        # 🔁 Start loader processes
        workers = []
        ready_to_read = multiprocessing.Event()
        setup_start_time = time.time()

        # Launch loader processes
        for worker_id in range(num_loader_workers):
            p = multiprocessing.Process(
                target=loader,
                args=(
                    worker_id,
                    input_queue,
                    output_queue,
                    error_counter,
                    error_lock,
                    ready_to_read,
                    len(jobs),
                    error_threshold,
                    setup_start_time,
                    loop_no,
                ),
                daemon=True,
            )
            self.record_pid(p.pid)
            p.start()
            workers.append(p)



        # 🧵 Start storer thread
        storer_thread = Thread(
            target=storer,
            args=(
                output_queue,
                error_counter,
                error_lock,
                self.instructions["HDF5_file_path"],
                stats_list,
                self.controller,
                len(jobs),
                loop_no,
            ),
            daemon=True,
        )
        storer_thread.start()

        # 🧼 Wait for all loaders to finish
        for p in workers:
            p.join()

        # ✅ Signal the storer to stop
        output_queue.put(None)

        # 🧼 Wait for storer to finish
        storer_thread.join()

        return pd.DataFrame(list(stats_list))



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
        avg_segmentation_time=('avg_segmentation_time', 'mean'),
        avg_preprocessing_time=('avg_preprocessing_time', 'mean'),
        avg_feature_extraction_time=('avg_feature_extraction_time', 'mean'),
    )
    summary_df["total_overhead"] = setup_ratio

    # Round elapsed time columns
    time_cols = ['elapsed_total', 'elapsed_seg', 'elapsed_binning',
                 'elapsed_store_image', 'elapsed_store_results',
                 'avg_segmentation_time', 'avg_preprocessing_time', 
                 'avg_feature_extraction_time']
    for col in time_cols:
        summary_df[col] = summary_df[col].map(lambda x: round(x, 2))

    # Compute rates
    summary_df['images_per_s'] = (summary_df['images'] * num_workers / summary_df['elapsed_total']).round(2)
    summary_df['stacks_per_s'] = (summary_df['jobs'] * num_workers / summary_df['elapsed_total']).round(2)
    summary_df['overhead_per_image'] = (summary_df["total_overhead"] / summary_df['images']).round(2)

    return summary_df