from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import logging

from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.models.production.orm.WoodTableA import WoodTableA

class WoodTableABase(BaseModel):
    family: Optional[str]
    genus: Optional[str]
    species: Optional[str]
    IFAW_ID: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]
    origin: Optional[str]
    sourceNo: Optional[str]

class WoodTableAOut(WoodTableABase):
    id: int
    class Config:
        orm_mode = True

    @classmethod
    def persist_to_db(cls, records: list["WoodTableAOut"], db_key="production"):
        session: Session = DBEngine(db_key).get_session()
        try:
            objs = [WoodTableA(**rec.model_dump(exclude_unset=True)) for rec in records]
            session.bulk_save_objects(objs)
            session.commit()
            logging.debug3(f"✅ Persisted {len(objs)} entries to WoodTableA.")
        except Exception as e:
            session.rollback()
            logging.error(f"❌ Failed to persist WoodTableA entries: {e}", exc_info=True)
            raise
        finally:
            session.close()
