from typing import List, Type, TypeVar
from sqlalchemy.orm import Session
from app.utils.SQL.DBEngine import DBEngine
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging

T = TypeVar("T", bound="SharedBaseModel")

class api_BaseModel(BaseModel):
    """
    Shared base for all Pydantic models in the application.
    """
    class Config:
        orm_mode = True
        anystr_strip_whitespace = True
        use_enum_values = True
        extra = "forbid"

    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert to dict."""
        return self.model_dump(exclude_none=exclude_none)

    def json_api(self, **kwargs: Any) -> str:
        """Pretty JSON for APIs/logs/debugging."""
        return self.model_dump_json(indent=2, exclude_none=True, **kwargs)
    
    @classmethod
    def store_dataframe(
        cls: Type[T], df: pd.DataFrame, orm_class: Type, session: Session
    ) -> None:
        """Validate and store a DataFrame using the ORM class and session."""
        try:
            validated = cls.from_dataframe(df)
            orm_objs = [item.to_orm(orm_class) for item in validated]
            session.bulk_save_objects(orm_objs)
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"❌ Failed to store data for {cls.__name__}: {e}", exc_info=True)
            raise
        finally:
            session.close()


    @classmethod
    def fetch_all(cls: Type[T], orm_class: Type, db_key: str = "raw") -> List[T]:
        """Fetch all records for the given ORM class and return them as Pydantic models."""
        session: Session = DBEngine(db_key).get_session()
        try:
            results = session.query(orm_class).all()
            return [cls.model_validate(r) for r in results]
        except Exception as e:
            logging.error(f"❌ Failed to fetch {orm_class.__name__} entries: {e}", exc_info=True)
            return []
        finally:
            session.close()

