import logging
import os
import pandas as pd
from typing import List
import yaml
import random
import time
import logging
from datetime import datetime

from sqlalchemy.orm import object_session

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.SQL.SQL_Dict import SQL_Dict
from app.utils.SQL.models.temp.api.api_DoEJobs import DoEJobs_Out
from app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.SQL.models.production.api.api_WoodTableB import WoodTableB_Out


from app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out
from app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out



from app.utils.SQL.models.production.api.api_ModellingResults import ModellingResults_Out


from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.dataModels.FilterModel.FilterModel import Border
from app.utils.dataModels.Jobs.DoEJob import DoEJob


from app.tasks.TA30_JobBuilder.TA30_A_ProviderJobBuilder import TA30_A_ProviderJobBuilder
from app.tasks.TA30_JobBuilder.TA30_B_SegmenterJobBuilder import TA30_B_SegmenterJobBuilder
from app.tasks.TA30_JobBuilder.TA30_C_ModelerJobBuilder import TA30_C_ModelerJobBuilder


#from app.utils.SQL.models.temp.api.SegmentationJobs_out import SegmentationJobs_out


class TA30_0_JobBuilderWrapper(TaskBase):
    def setup(self):
        self.controller.update_message("Initializing Segmentation Job Builder")
        self.controller.update_progress(0.01)
        self.woodmaster_df = pd.DataFrame()
        self.general_job_df = pd.DataFrame()
        self.filtered_jobs_df = pd.DataFrame()
        self.segmentation_jobs = []
        logging.info("[TA30_A] Setup complete.")



    def run(self):
        try:
            logging.info("[TA30] Starting infinite job builder loop")
            while True:
                loop_start = datetime.utcnow()
                logging.info(f"[TA30] Job loop started at {loop_start.isoformat()}")

                self.controller.update_message("Scanning for new DoE Jobs")
                builders = ["provider", "segmenter", "modeler"]

                for b in builders:
                    status_col = f"{b}_status"

                    include_cols = [
                        "sourceNo", "woodType", "family", "genus", "species",
                        "view", "lens", "noShotsRange", "maxShots", "filterNo"
                    ]


                    if "totalNumberShots" not in include_cols:
                        include_cols.append("totalNumberShots")
                    if "noShotsRange" in include_cols:
                        include_cols.remove("noShotsRange")

                    match b:
                        case "provider":
                            BuilderClass = TA30_A_ProviderJobBuilder
                            groupby_col = "sampleID"
                            id_field = "job_uuid"
                            if "maxShots" in include_cols:
                                include_cols.remove("maxShots")
                            if "filterNo" in include_cols:
                                include_cols.remove("filterNo")



                            filter_table = WoodMasterPotential_Out
                        case "segmenter":
                            groupby_col = "stackID"

                            BuilderClass = TA30_B_SegmenterJobBuilder
                            filter_table = WoodMaster_Out
                        case "modeler":
                            BuilderClass = TA30_C_ModelerJobBuilder

                    self.controller.update_message(f"Checking {b} jobs (status: todo)")
                    filter_model = FilterModel.from_human_filter({"contains": {status_col: "todo"}})
                    raw_df = DoEJobs_Out.fetch(filter_model=filter_model)

                    if raw_df.empty:
                        logging.debug(f"[TA30] No '{b}' jobs found.")
                        continue

                    jobs = raw_df["payload"].apply(DoEJob.model_validate).tolist()
                    logging.info(f"[TA30] {len(jobs)} '{b}' jobs found in DB.")

                    job_df = self.expand_jobs_via_filters(
                        jobs,
                        include_cols=include_cols,
                        is_range_cols=["totalNumberShots"],
                        is_max=["maxShots"],
                        src_data_api=filter_table,
                        id_field=id_field,
                    )

                    if job_df.empty:
                        logging.warning(f"[TA30] No stack rows matched for {b} jobs.")
                        continue

                    logging.info(f"[TA30] Dispatching {len(job_df)} rows to {BuilderClass.__name__}")
                    BuilderClass.build(job_df)

                self.controller.update_message("Sleeping")
                logging.info("[TA30] Sleeping for 3 minutes to allow other tasks to process.")
                time.sleep(180)

        except KeyboardInterrupt:
            logging.info("[TA30] Interrupted by user — shutting down gracefully.")
            self.controller.finalize_failure("Interrupted by user")
        except Exception as e:
            logging.exception("[TA30] JobBuilder task failed", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()



    def cleanup(self):
        logging.info("[TA30_A] Running cleanup routine.")
        self.controller.archive_with_orm()


    def expand_jobs_via_filters(
        self,
        jobs: list,
        *,
        include_cols: list[str],
        is_range_cols: list[str] = None,
        is_max: list[str] = None,
        is_min: list[str] = None,
        border_rule: dict = None,
        global_logic: str = "and",
        src_data_api: WoodMaster_Out,
        id_field="job_uuid",
        groupby_col: str = "sampleID"
    ) -> pd.DataFrame:
        """
        Expand a list of Job models (e.g., DoEJob) into a long-form dataset
        using per-job FilterModels and SQL table fetching.

        Returns:
            DataFrame containing all matched rows with originating job_uuid
        """
        filter_df = jobs[0].__class__.to_filter_df(jobs)
        filter_df = filter_df.reset_index().rename(columns={"index": id_field})
        job_filters = FilterModel.from_dataframe(
            df=filter_df,
            include_cols=include_cols,
            is_range_cols=is_range_cols,
            is_max=is_max,
            is_min=is_min,
            border_rule=border_rule,
            global_logic=global_logic,
            job_id_field=id_field
        )

        dtypes = src_data_api.pydantic_model_to_dtype_dict()
        new_subset = None

        result_df = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
        total_jobs = len(job_filters)

        for i, filter_model in enumerate(job_filters):
            
            
            if i < 350:
                 continue  # Skip first 385 jobs for testing purposes
            if i % 10 == 0 or i == total_jobs - 1:
                logging.debug2(f"[Expand] Processing job {i+1}/{total_jobs}")
            
            del new_subset
            new_subset = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
            new_subset = new_subset.copy()
            with self.suppress_logging():
                new_subset = src_data_api.fetch(filter_model=filter_model, stream=False)
            
            new_subset["og_job_uuids"] = filter_model.job_id
            new_subset = new_subset.copy()
            len_new = len(new_subset)
            len_before = len(result_df)

            if not new_subset.empty:
                with self.suppress_logging():
                    result_df = pd.concat([result_df, new_subset], ignore_index=True)

            len_after = len(result_df)
            delta = len_after - len_before

            if i % 10 == 0 or i == total_jobs - 1:
                ratio = delta / len_new if len_new > 0 else "N/A"
                logging.debug2(
                    f"[Expand] Job {i+1}/{total_jobs} Old: {len_before}, New: {len_new}, "
                    f"Combined: {len_after}, Added Ratio: {ratio}"
                )


        if not result_df.empty:
            group_cols = [col for col in result_df.columns if col != "og_job_uuids"]
            result_df["og_job_uuids"] = result_df["og_job_uuids"].apply(lambda x: [x])

            result_df = (
                result_df.groupby(groupby_col, as_index=False)
                .agg({
                    **{col: "first" for col in group_cols if col != groupby_col},
                    "og_job_uuids": lambda x: sorted(set(sum(x, [])))
                })
            )

        return result_df


    @staticmethod
    def store_and_update(to_create, to_update, orm_instance):
        """
        Stores new jobs and updates the og_job_uuids field of existing ones.

        Args:
            to_create: list[ORM] — fresh jobs to insert
            to_update: list[ORM] — existing jobs that need parent list update
            orm_instance: ORM class (e.g., ProviderJobs)
        """

        if not to_create and not to_update:
            logging.info("No jobs to create or update.")
            return

        from app.utils.SQL.DBEngine import DBEngine
        session = DBEngine("temp").get_session()


        created_count = 0
        updated_count = 0
        unchanged_count = 0
        update_log_lines = []

        # INSERT new jobs
        if to_create:
            session.add_all(to_create)
            created_count = len(to_create)

        # UPDATE existing jobs
        if to_update:
            incoming_by_uuid = {job.job_uuid: job for job in to_update}

            db_jobs = session.query(orm_instance).filter(
                orm_instance.job_uuid.in_(incoming_by_uuid.keys())
            ).all()

            for db_job in db_jobs:
                incoming_job = incoming_by_uuid[db_job.job_uuid]
                old_parents = set(db_job.og_job_uuids or [])
                new_parents = set(incoming_job.og_job_uuids or [])

                combined = old_parents | new_parents

                if combined != old_parents:
                    db_job.og_job_uuids = list(combined)
                    updated_count += 1
                    update_log_lines.append(
                        f"→ Job {db_job.job_uuid} updated:\n"
                        f"   old: {sorted(old_parents)}\n"
                        f"   new: {sorted(combined)}\n"
                    )
                else:
                    unchanged_count += 1

        session.commit()

        # Consolidated logging message
        logging.info(
            f"Job store/update summary of: \n"
            f"{orm_instance.__name__} \n" 
            f"  ✅ Created: {created_count}\n"
            f"  🔁 Updated: {updated_count}\n"
            f"  ⏭️ Unchanged: {unchanged_count}\n"
        )
        logging.debug2(
            f"{''.join(update_log_lines) if update_log_lines else ''}"
            f"  ✅ Session committed.\n"
        )














