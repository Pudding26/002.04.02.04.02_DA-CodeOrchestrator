import logging

from typing import List, Dict, Any
import pandas as pd
import os


from app.utils.dataModels.Jobs.ModelerJob import ModelerJob, ModelerJobInput, ModelerAttrs
from app.utils.dataModels.Jobs.JobEnums import JobStatus

from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out
from app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out



def bulk_update_status() -> int:
    """
    Marks jobs as DONE if their parent_job_uuid exists in ModellingResults.

    Parameters
    ----------
    job_uuids : list
        List of job UUIDs to check.

    Returns
    -------
    int
        Number of jobs marked as DONE.
    """

    possible_jobs = check_for_jobs("modeler")
    possible_jobs_list = possible_jobs["job_uuid"].tolist()


    filter_model = FilterModel.from_human_filter(
        {"contains": {"job_uuid": {"or": possible_jobs_list}}},
    )

    job_df = WorkerJobs_Out.fetch(filter_model=filter_model, stream=False)
    job_list = convert_raw_dataframe(job_df)

    # Fetch distinct UUIDs from results table
    done_jobs = ModellingResults_Out.fetch_distinct_values(column="DoE_UUID")

    already_done = 0
    for i, job in enumerate(job_list):
        try:
            if job.parent_job_uuids and job.parent_job_uuids[0] in done_jobs:
                if i % 100 == 0 or i == len(job_list):
                    logging.debug2(f"[MARKER] Job {job.job_uuid} already processed, setting status DONE.")
                job.status = JobStatus.DONE
                job.update_db(fields_to_update=["status", "payload"])
                already_done += 1
        except Exception as e:
            logging.warning(f"[MARKER WARNING] Could not update job {job.job_uuid}: {str(e)}")

    logging.debug5(f"[MARKER] Marked {already_done} jobs as DONE")
    return already_done


def check_for_jobs(job_type):
    """
    Query DB for jobs of a given type that are READY to be scheduled.
    Returns a pandas DataFrame of jobs.
    """
    filter_model = FilterModel.from_human_filter({
        "contains": {"status": "READY", "job_type": job_type}
    })
    return WorkerJobs_Out.fetch(filter_model=filter_model, stream=False)

def convert_raw_dataframe(job_df_raw: pd.DataFrame) -> List[ModelerJob]:
    jobs = []
    total_jobs = len(job_df_raw)
    log_interval = 10

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



    for i, (_, row) in enumerate(job_df_raw.iterrows()):
        try:
            payload: Dict[str, Any] = row["payload"]
            inp = ModelerJobInput(
                stackIDs=payload["input"]["stackIDs"],
                preProcessing_instructions=payload["input"].get("preProcessing_instructions", {}),
                metricModel_instructions=payload["input"].get("metricModel_instructions", {}),
                preProcessingNo=payload["input"]["preProcessingNo"],
                metricModelNo=payload["input"]["metricModelNo"],
                scope=payload["input"]["scope"],
            )
            job = ModelerJob(
                job_uuid=row["job_uuid"],
                job_type=row["job_type"],
                status=row["status"],
                attempts=row["attempts"],
                next_retry=row["next_retry"],
                created=row["created"],
                updated=row["updated"],
                parent_job_uuids=row["parent_job_uuids"],
                input=inp,
                attrs=ModelerAttrs(
                    preprocessed_data=None,  # Will be set later
                    raw_data=None,  # Will be set later
                    model_results=None  # Will be set later
                )
            )
            jobs.append(job)

            if i % log_interval == 0 or i == total_jobs - 1:
                logging.debug2(f"[LOADER] Parsed {i+1}/{total_jobs} jobs")

        except Exception as e:
            logging.warning(f"[LOADER WARNING] Failed to parse row {i}: {str(e)}")

    return jobs