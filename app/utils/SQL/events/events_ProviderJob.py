from typing import Any

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper


from app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.dataModels.Jobs.JobLink import JobLink
from app.utils.dataModels.Jobs.DoEJob import DoEJob

from app.utils.SQL.models.temp.orm.ProviderJobs import ProviderJobs


from app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState

from app.utils.SQL.models.temp.api.api_ProviderJobs import ProviderJobs_Out




def _status_to_rel_state(status: JobStatus) -> RelationState:
    """
    Maps a JobStatus (TODO, FAILED, DONE) to a RelationState for JobLink rows.
    
    Args:
        status (JobStatus): The current status of a child job.

    Returns:
        RelationState: The corresponding relation state used for linkage.
    """
    return (
        RelationState.IN_PROGRESS if status == JobStatus.TODO else
        RelationState.FAILED if status == JobStatus.FAILED else
        RelationState.FREE
    )


@event.listens_for(ProviderJobs, "after_insert")
@event.listens_for(ProviderJobs, "after_update")
def sync_provider_links(mapper: Mapper, conn: Connection, target: ProviderJobs) -> None:
    """
    Keeps the JobLink table in sync after a ProviderJob is inserted or updated.
    
    Steps:
        1. Deletes any existing JobLink entries for this child.
        2. Re-inserts links based on `og_job_uuids`.
        3. Recalculates the roll-up status of each linked parent DoEJob.
    
    Args:
        mapper (Mapper): SQLAlchemy mapper for ProviderJob.
        conn (Connection): Active DB connection (shared with the current session).
        target (ProviderJob): The ProviderJob instance that was changed.
    """
    # Step 1: Remove old link records
    conn.execute(
        delete(JobLink).where(
            JobLink.child_uuid == target.job_uuid,
            JobLink.child_kind == JobKind.PROVIDER
        )
    )

    # Step 2: Create new link rows based on og_job_uuids
    rows = [
        {
            "parent_uuid": str(pid),
            "child_uuid": target.job_uuid,
            "child_kind": JobKind.PROVIDER,
            "rel_state": _status_to_rel_state(target.status)
        }
        for pid in target.og_job_uuids
    ]

    if rows:
        conn.execute(insert(JobLink), rows)

    # Step 3: Trigger roll-up update for each parent DoEJob
    for pid in target.og_job_uuids:
        _roll_up_provider_status(conn, str(pid))


def _roll_up_provider_status(conn: Connection, parent_uuid: str) -> None:
    """
    Aggregates the state of all Provider child jobs for a DoEJob and updates the provider_status field.
    
    Rule:
        - If any child is TODO → status = TODO
        - Else if >30% children FAILED → status = FAILED
        - Else → status = DONE

    Args:
        conn (Connection): Active DB connection.
        parent_uuid (str): The str of the parent DoEJob whose status should be updated.
    """
    # Aggregate state from JobLink
    total, todo, failed = conn.execute(
        select(
            func.count(),
            func.sum(func.case((JobLink.rel_state == RelationState.IN_PROGRESS, 1), else_=0)),
            func.sum(func.case((JobLink.rel_state == RelationState.FAILED, 1), else_=0)),
        ).where(
            JobLink.parent_uuid == parent_uuid,
            JobLink.child_kind == JobKind.PROVIDER,
        )
    ).one()

    # Decide new status
    if total == 0:
        new_status = JobStatus.DONE
    elif todo > 0:
        new_status = JobStatus.TODO
    elif failed / total > 0.3:
        new_status = JobStatus.FAILED
    else:
        new_status = JobStatus.DONE

    # Update parent DoEJob with new status
    conn.execute(
        update(DoEJob)
        .where(DoEJob.job_uuid == parent_uuid)
        .values(provider_status=new_status)
    )
