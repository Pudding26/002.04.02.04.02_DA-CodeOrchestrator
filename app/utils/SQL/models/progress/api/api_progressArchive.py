from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.utils.SQL.models.progress.orm.ProgressArchive import ProgressArchive
from app.utils.SQL.DBEngine import DBEngine  # Assuming you have a helper

class ProgressArchiveBase(BaseModel):
    task_uuid: str
    task_name: str
    start_time: Optional[datetime]
    finish_time: Optional[datetime]
    status: Optional[str]
    finished: Optional[str]
    message: Optional[str]
    progress: Optional[float]
    elapsed_time: Optional[float]
    total_size: Optional[float]
    data_transferred_gb: Optional[float]
    item_count: Optional[int]
    stack_count: Optional[int]


class ProgressArchiveOut(ProgressArchiveBase):
    class Config:
        orm_mode = True

    @classmethod
    def persist_to_db(cls, data: "ProgressArchiveOut"):
        """Stores the archive entry in the database."""
        engine = DBEngine("progress")
        session: Session = engine.get_session()

        try:
            entry = ProgressArchive(**data.dict())
            session.add(entry)
            session.commit()
            return entry
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to persist ProgressArchive: {e}")
        finally:
            session.close()
