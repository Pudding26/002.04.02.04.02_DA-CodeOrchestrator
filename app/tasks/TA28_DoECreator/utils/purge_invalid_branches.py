# app/tasks/TA28_DoECreator/utils/purge_invalid_branches.py

import yaml
import logging
from pathlib import Path
from sqlalchemy import select, update
from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
from app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink
from app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs

PURGE_FILE = Path("tmp/purge_DEAP_branches.yaml")


def purge_invalid_branches():
    """
    Purge jobs/control subbranches listed in PURGE_FILE and reset system.

    Steps:
    1. For each subbranch in purge_branches.yaml:
        - Remove the subbranch from all DEAP_metadata[branch].
        - If a branch has no subbranches left, drop that branch.
        - If metadata becomes empty, delete the job.
    2. Clear WorkerJobs and JobLinks.
    3. Reset all DoEJobs statuses:
        provider_status = READY
        all other statuses = BLOCKED
    4. Clear PURGE_FILE after success.
    """

    if not PURGE_FILE.exists():
        PURGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PURGE_FILE, "w") as f:
            yaml.safe_dump([], f)

    with open(PURGE_FILE, "r") as f:
        subbranches_to_purge = yaml.safe_load(f) or []

    if not subbranches_to_purge:
        return 0, 0

    logging.info(f"🧹 Purging control subbranches: {subbranches_to_purge}")
    session = DBEngine(db_key="jobs").get_session()

    deleted_jobs = 0
    updated_jobs = 0

    try:
        jobs = session.execute(select(orm_DoEJobs)).scalars().all()

        for job in jobs:
            metadata = job.DEAP_metadata or {}
            modified = False

            # iterate over top-level branches
            for branch in list(metadata.keys()):
                subbranches = metadata.get(branch, {})
                if not isinstance(subbranches, dict):
                    continue

                # remove matching subbranches
                for subbranch in list(subbranches.keys()):
                    if subbranch in subbranches_to_purge:
                        logging.debug(f"✂️ Removing subbranch {subbranch} from branch {branch} in job {job.job_uuid}")
                        subbranches.pop(subbranch, None)
                        modified = True

                # if branch has no subbranches left → remove branch
                if not subbranches:
                    logging.debug(f"🗑️ Removing empty branch {branch} from job {job.job_uuid}")
                    metadata.pop(branch, None)
                    modified = True

            # delete whole job if metadata empty
            if not metadata and modified:
                logging.debug(f"🗑️ Deleting job {job.job_uuid} (all branches removed)")
                session.delete(job)
                deleted_jobs += 1
            elif modified:
                session.execute(
                    update(orm_DoEJobs)
                    .where(orm_DoEJobs.job_uuid == job.job_uuid)
                    .values(DEAP_metadata=metadata)
                )
                updated_jobs += 1

        # --- reset system state ---



        if deleted_jobs == 0 and updated_jobs == 0:
            logging.info("No jobs modified or deleted. Nothing to purge.")
            return 0, 0
        logging.info("🧨 Resetting WorkerJobs and JobLinks tables...")
        session.query(orm_WorkerJobs).delete()
        session.query(orm_JobLink).delete()


        #logging.info("🔄 Resetting DoEJobs statuses...")
        #session.execute(
        #    update(orm_DoEJobs).values(
        #        provider_status="READY",
        #        segmenter_status="BLOCKED",
        #        extractor_status="BLOCKED",
        #        modeler_status="BLOCKED",
        #        validator_status="BLOCKED",
        #    )
        #)


        session.commit()

        # clear purge file
        with open(PURGE_FILE, "w") as f:
            yaml.safe_dump([], f)

        logging.info(f"✅ Purge completed. Deleted={deleted_jobs}, Updated={updated_jobs}")
        return deleted_jobs, updated_jobs

    except Exception as e:
        session.rollback()
        logging.error(f"❌ Branch purge failed: {e}", exc_info=True)
        raise
    finally:
        session.close()
