# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_DS09Entry.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from sqlalchemy.orm import Session
import logging

from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.models.production.orm.DS09 import DS09

class DS09Out(BaseModel):
    
    id: int
    IFAW_code: Optional[str]
    origin: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    species: Optional[str]
    genus: Optional[str]

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def fetch_all(cls, db_key="production") -> List["DS09"]:
        session: Session = DBEngine(db_key).get_session()
        try:
            rows = session.query(DS09).all()
            return [cls.model_validate(row) for row in rows]
        except Exception as e:
            logging.error(f"❌ Failed to fetch DS09 entries: {e}", exc_info=True)
            return []
        finally:
            session.close()
