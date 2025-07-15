def store_or_update_jobs(job_df):
    """
    Store or update a batch of DoE jobs in the jobs database.

    For each row in `job_df`:
    - If `job_uuid` does not exist in DB ⇒ create new row.
    - If `job_uuid` exists ⇒ merge `DEAP_metadata` and update row.

    Returns:
        tuple:
            - created_count: number of new rows inserted
            - updated_count: number of existing rows updated
    """
    from app.utils.SQL.DBEngine import DBEngine
    from app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
    from sqlalchemy import select, update
    import logging

    session = DBEngine(db_key="jobs").get_session()

    created_count = 0
    updated_count = 0

    try:
        for job in job_df.to_dict(orient="records"):
            job_uuid = job["job_uuid"]

            # Check if job already exists
            existing = session.execute(
                select(orm_DoEJobs).where(orm_DoEJobs.job_uuid == job_uuid)
            ).scalar_one_or_none()

            if not existing:
                session.add(orm_DoEJobs(**job))
                created_count += 1
            else:
                existing_metadata = existing.DEAP_metadata or {}
                new_metadata = job.get("DEAP_metadata", {})

                # Merge logic: deep update per branch
                for branch, subbranches in new_metadata.items():
                    if branch not in existing_metadata:
                        existing_metadata[branch] = subbranches
                    else:
                        existing_metadata[branch].update(subbranches)

                session.execute(
                    update(orm_DoEJobs)
                    .where(orm_DoEJobs.job_uuid == job_uuid)
                    .values(DEAP_metadata=existing_metadata)
                )
                updated_count += 1

        session.commit()

    except Exception as e:
        session.rollback()
        logging.error(f"❌ Failed to store/update DoE_Jobs: {e}", exc_info=True)
        raise e

    finally:
        session.close()

    return created_count, updated_count
