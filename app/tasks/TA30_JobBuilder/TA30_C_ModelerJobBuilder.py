import logging
import os
import pandas as pd
from typing import List
import yaml
import random
import hashlib


from app.tasks.TaskBase import TaskBase


from app.utils.SQL.models.jobs.api_DoEJobs import DoEJobs_Out
from app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out
from app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out



from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.dataModels.FilterModel.FilterModel import Border

from app.utils.dataModels.Jobs.DoEJob import DoEJob
from app.utils.dataModels.Jobs.ModelerJob import ModelerJob, ModelerJobInput, ModelerAttrs, PreProcessingAttributes, MetricModelAttributes

from app.utils.dataModels.Jobs.JobEnums import JobKind, JobStatus

from app.utils.dataModels.Jobs.ModelerJob import (
    ModelerJob, ModelerJobInput
)



#from app.utils.SQL.models.temp.api.SegmentationJobs_out import SegmentationJobs_out


class TA30_C_ModelerJobBuilder:
    """Build and persist ModelerJobs (mirrors ProviderJobBuilder)."""

    @classmethod
    def build(cls, job_df: pd.DataFrame, jobs) -> None:
        if job_df.empty:
            logging.info("[ModelerJobBuilder] Nothing to build.")
            return

        job_df_raw = job_df.copy()
        logging.info("CWD =", os.getcwd())
        
        with open("app/config/presets/preProcessingPresets.yaml") as file:
            PREP_PRESETS = yaml.safe_load(file)

        with open("app/config/presets/metricModelPresets.yaml") as file:
            MM_PRESETS = yaml.safe_load(file)

        uuid_to_metricModelNo = {
            job.job_uuid: job.doe_config.modeling.metricModelNo[0]
            for job in jobs
        }
        uuid_to_preProcessingNo = {
            job.job_uuid: job.doe_config.preprocessing.preProcessingNo[0]
            for job in jobs
        }
         

        # 1) Explode so each parent_job_uuid is its own row
        exploded_df = job_df_raw.explode("parent_job_uuids").rename(columns={"parent_job_uuids": "job_uuid"})

        # 2) Merge the dest_FilterNo from the DoE jobs
        # Here, uuid_to_filter is a dict mapping parent DoE job UUIDs → FilterNo
        exploded_df = exploded_df[["job_uuid", "stackID", "path"]]
        exploded_df["metricModelNo"] = exploded_df["job_uuid"].map(uuid_to_metricModelNo)
        exploded_df["preProcessingNo"] = exploded_df["job_uuid"].map(uuid_to_preProcessingNo)

        # 3) Group by stackID (e.g. sampleID) AND dest_FilterNo
        group_cols = ["job_uuid"]  # add any other cols you need

        agg_df = exploded_df.groupby(group_cols, as_index=False).agg(
            {
                **{c: "first" for c in exploded_df.columns if c not in group_cols + ["stackID", "path"]},
                "stackID": list,  # list of all parent ids
                "path": list  # list of all parent ids
            }
        )




        agg_df["parent_job_uuids"] = agg_df["job_uuid"]


        agg_df["job_uuid"] = agg_df["parent_job_uuids"].str.split("_").str[1].apply(
            lambda v: "modeler_" + hashlib.sha1(str(v).encode()).hexdigest()[:10]
        )

        existing = WorkerJobs_Out.fetch_distinct_values(column="job_uuid")


        to_create: List[ModelerJob] = []
        to_update: List[ModelerJob] = []


        logging.debug3(f"Starting to create a total of ModelerJobs", len(job_df))
        def find_scope_in_dict(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == "scope" and v is not None:
                        return v
                    if isinstance(v, (dict, list)):
                        found = find_scope_in_dict(v)
                        if found is not None:
                            return found
            elif isinstance(d, list):
                for item in d:
                    found = find_scope_in_dict(item)
                    if found is not None:
                        return found
            return None


        for jobNo, row in agg_df.iterrows():
            metricModelNo = row.get("metricModelNo")
            preProcessingNo = row.get("preProcessingNo")

            job = ModelerJob(
                job_uuid=row["job_uuid"],
                parent_job_uuids=[row.get("parent_job_uuids")],
                status=JobStatus.READY.value,
                job_type=JobKind.MODELER.value,
                input=ModelerJobInput(
                    stackIDs=row["stackID"],
                    preProcessingNo=preProcessingNo,
                    metricModelNo=metricModelNo,
                    job_No=jobNo,
                    preProcessing_instructions=PreProcessingAttributes(**PREP_PRESETS.get(preProcessingNo, {})),
                    metricModel_instructions=MetricModelAttributes(**MM_PRESETS.get(metricModelNo, {})),
                    scope=find_scope_in_dict(d =PREP_PRESETS)
                ),
                attrs=ModelerAttrs(),  # default empty
            )



            if job.job_uuid in existing:
                to_update.append(job)
            else:
                to_create.append(job)

            if jobNo % 100 == 0 or jobNo == len(job_df) - 1:
                logging.debug2(
                    "Processed %d/%d jobs: %s",
                    jobNo + 1,
                    len(job_df),
                    job.job_uuid
                )

        logging.info(
            "[ModelerJobBuilder] New: %d, Update: %d, Total: %d",
            len(to_create),
            len(to_update),
            len(to_create) + len(to_update),
        )


        from app.tasks.TA30_JobBuilder.TA30_0_JobBuilderWrapper import TA30_0_JobBuilderWrapper
        TA30_0_JobBuilderWrapper.store_and_update(
            to_create=to_create, to_update=to_update
        )


