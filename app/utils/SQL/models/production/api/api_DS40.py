# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_DS09Entry.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from sqlalchemy.orm import Session
import logging

from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.models.production.orm.DS40 import DS40

class DS40Out(BaseModel):
    
    genus: int
    family: str


    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def fetch_all(cls, db_key="production") -> List["DS40"]:
        session: Session = DBEngine(db_key).get_session()
        try:
            rows = session.query(DS40).all()
            return [cls.model_validate(row) for row in rows]
        except Exception as e:
            logging.error(f"❌ Failed to fetch DS40 entries: {e}", exc_info=True)
            return []
        finally:
            session.close()
