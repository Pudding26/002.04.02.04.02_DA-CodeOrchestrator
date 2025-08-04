from __future__ import annotations
import logging, json
import pandas as pd
from typing import List
from uuid import uuid4
from datetime import datetime
import hashlib
import yaml
import os

from sqlalchemy.orm import Session

from app.utils.general.HelperFunctions import add_hashed_uuid_column


from app.utils.dataModels.Jobs.JobEnums import JobKind, JobStatus

from app.utils.dataModels.Jobs.SegmenterJob import (
    SegmenterJob, SegmenterJobInput
)



from app.utils.SQL.models.jobs.api_DoEJobs import DoEJobs_Out
from app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out
from app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out




#from app.utils.SQL.models.temp.api.SegmentationJobs_out import SegmentationJobs_out


class TA30_B_SegmenterJobBuilder:
    """Build and persist SegmenterJobs (mirrors ProviderJobBuilder)."""

    @classmethod
    def build(cls, job_df: pd.DataFrame, jobs) -> None:
        if job_df.empty:
            logging.info("[SegmenterJobBuilder] Nothing to build.")
            return

        job_df_raw = job_df.copy()
        job_df_raw["attrs_raw"] = job_df_raw.apply(
            lambda row: row.dropna().to_dict(),
            axis=1
        )

        uuid_to_dest_filter = {
            job.job_uuid: job.doe_config.segmentation.filterNo[0]
            for job in jobs
        }



        # 1) Explode so each parent_job_uuid is its own row
        exploded_df = job_df_raw.explode("parent_job_uuids").rename(columns={"parent_job_uuids": "job_uuid"})

        # 2) Merge the dest_FilterNo from the DoE jobs
        # Here, uuid_to_filter is a dict mapping parent DoE job UUIDs → FilterNo
        exploded_df["dest_FilterNo"] = exploded_df["job_uuid"].map(uuid_to_dest_filter)

        # 3) Group by stackID (e.g. sampleID) AND dest_FilterNo
        group_cols = ["sampleID", "dest_FilterNo"]  # add any other cols you need
        agg_df = exploded_df.groupby(group_cols, as_index=False).agg(
            {
                **{c: "first" for c in exploded_df.columns if c not in group_cols + ["job_uuid"]},
                "job_uuid": list  # list of all parent ids
            }
        )
        agg_df = agg_df.rename(columns={"job_uuid": "parent_job_uuids"})

        job_df = agg_df.copy()


        job_df["dest_stackID_FF"] = job_df["stackID"].apply(
            lambda x: "_".join(x.split("_")[:-1])) + "_" + job_df["dest_FilterNo"].astype(str)
        
        job_df["dest_stackID_GS"] = job_df["stackID"].apply(
            lambda x: "_".join(x.split("_")[:-1]) + "_" + "GS")

        job_df["job_uuid"] = job_df["dest_stackID_FF"].apply(
            lambda v: "segmenter_" + hashlib.sha1(str(v).encode()).hexdigest()[:10]
        )
        job_df["dest_file_path_FF"] = job_df.apply(
            lambda row: os.path.join(
            "/".join(row["path"].split("/")[:-1]), row["dest_stackID_FF"]
            ),
            axis=1
        )

        job_df["dest_file_path_GS"] = job_df.apply(
            lambda row: os.path.join(
            "/".join(row["path"].split("/")[:-1]), row["dest_stackID_GS"]
            ),
            axis=1
        )




        existing = WorkerJobs_Out.fetch_distinct_values(column="job_uuid")


        to_create: List[SegmenterJob] = []
        to_update: List[SegmenterJob] = []


        with open("app/config/segmentationFilter.yaml") as file:
            FILTER_PRESETS = yaml.safe_load(file)

        logging.debug3(f"Starting to create a total of %d SegmenterJobs", len(job_df))

        for jobNo, row in job_df.iterrows():
            filter_no = row.get("dest_FilterNo")
            filter_instructions = FILTER_PRESETS.get(filter_no, {})

            job = SegmenterJob(
                job_uuid=row["job_uuid"],
                parent_job_uuids=row.get("parent_job_uuids", []),
                status=JobStatus.READY.value,
                job_type=JobKind.SEGMENTER.value,
                input=SegmenterJobInput(
                    src_file_path=row["path"],
                    dest_file_path_GS=row["dest_file_path_GS"],
                    dest_file_path_FF=row["dest_file_path_FF"],
                    dest_stackID_FF=row["dest_stackID_FF"],
                    dest_stackID_GS=row["dest_stackID_GS"],
                    dest_FilterNo=row["dest_FilterNo"],
                    filter_instructions=FILTER_PRESETS.get(filter_no, {}),
                ),
                attrs={"attrs_raw": row["attrs_raw"]},
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
            "[SegmenterJobBuilder] New: %d, Update: %d, Total: %d",
            len(to_create),
            len(to_update),
            len(to_create) + len(to_update),
        )


        from app.tasks.TA30_JobBuilder.TA30_0_JobBuilderWrapper import TA30_0_JobBuilderWrapper
        TA30_0_JobBuilderWrapper.store_and_update(
            to_create=to_create, to_update=to_update
        )



        return {
            "created": len(to_create),
            "updated": len(to_update)
        }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    