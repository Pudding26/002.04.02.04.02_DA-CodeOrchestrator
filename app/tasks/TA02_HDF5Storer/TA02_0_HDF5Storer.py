from app.tasks.TaskBase import TaskBase

from app.tasks.TA02_HDF5Storer.shared_queue import store_queue
from app.utils.dataModels.Jobs.SWMR_StorerJob import SWMR_StorerJob
from app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler
from app.utils.HDF5.HDF5_Inspector import HDF5Inspector

from queue import Empty
import logging
from datetime import datetime, timezone

class TA02_0_HDF5Storer(TaskBase):
    def setup(self):
        pass


    def run(self):
        hdf5_path = self.instructions["HDF5_file_path"]
        logging.info(f"🧵 [Storer] Starting SWMR storer on {hdf5_path}")

        try:
            handler = SWMR_HDF5Handler(file_path=hdf5_path)

            while True:
                if self.controller.should_stop():
                    logging.info("🛑 [Storer] Task stopped by controller")
                    self.controller.finalize_success()
                    break

                try:
                    job: SWMR_StorerJob = store_queue.get(timeout=2)
                except Empty:
                    continue

                try:
                    if job.handler_method == "handle_dataset":
                        handler.handle_dataset(
                            hdf5_file=handler.file,
                            dataset_name=job.dataset_path,
                            numpy_array=job.image_data,
                            attributes_new=job.attributes,
                            attribute_process=job.attribute_process,
                        )
                    else:
                        handler.store_image(
                            dataset_path=job.dataset_path,
                            image_data=job.image_data,
                            attributes=job.attributes,
                        )

                    HDF5Inspector.update_woodMaster_paths(
                        hdf5_path=hdf5_path,
                        dataset_paths=job.dataset_path
                    )

                    logging.debug1(f"📦 Stored: {job.dataset_path}")
                    job.result_queue.put((True, "ok"))

                except Exception as e:
                    logging.exception(f"❌ [Storer] Failed to store {job.dataset_path}")
                    job.result_queue.put((False, e))
                finally:
                    store_queue.task_done()

        finally:
            logging.info("🧹 [Storer] Cleanup")
            self.cleanup()

    def cleanup(self):
        logging.debug2("🧹 Running cleanup")
        self.controller.archive_with_orm()
