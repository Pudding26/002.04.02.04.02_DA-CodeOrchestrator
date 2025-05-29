from typing import List, Type, TypeVar, Optional, Any
from sqlalchemy.orm import Session
from app.utils.SQL.DBEngine import DBEngine
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from pydantic import TypeAdapter


T = TypeVar("T", bound="SharedBaseModel")

class api_BaseModel(BaseModel):
    """
    Shared base for all Pydantic models in the application.
    """
    class Config:
        from_attributes = True
        str_strip_whitespace = True
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
    def store_dataframe(cls, df: pd.DataFrame, db_key: str, method: str = "append") -> None:
        """
        Validate DataFrame using Pydantic v2 (vectorized), then store it via to_sql.
        Only supports 'append' or 'replace'.
        """
        from app.utils.SQL.DBEngine import DBEngine
        engine = DBEngine(db_key).get_engine()

        # ✅ Vectorized Pydantic validation
        try:
            validated_df = cls._model_validate_dataframe(df)
        except Exception as e:
            logging.error(f"❌ Pydantic validation failed in {cls.__name__}: {e}", exc_info=True)
            raise

        # ✅ Store to SQL
        try:
            validated_df.to_sql(
                name=cls.orm_class.__tablename__,
                con=engine,
                if_exists=method,
                index=False,
                method="multi",
                chunksize=10_000
            )
            logging.info(f"✅ Stored {len(validated_df)} rows to {cls.orm_class.__tablename__} using to_sql ({method})")
        except Exception as e:
            logging.error(f"❌ Failed to store to SQL in {cls.__name__}: {e}", exc_info=True)
            raise

    @classmethod
    def fetch_all(cls: Type[T], db_key: str = None, orm_class: Optional[Type] = None) -> List[T]:
        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")

        if orm_class is None:
            raise ValueError(f"{cls.__name__} must define 'orm_class' or pass it explicitly.")
        
        session = DBEngine(db_key).get_session()
        logging.debug2(f"🔍 Fetching all entries from {orm_class.__name__} in {db_key} database")
        try:
            results = session.query(orm_class).all()
            return [cls.model_validate(r) for r in results]
        except Exception as e:
            logging.error(f"❌ Failed to fetch {orm_class.__name__} entries: {e}", exc_info=True)
            return []
        finally:
            session.close()

    @classmethod
    def _model_validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        adapter = TypeAdapter(list[cls])
        validated = adapter.validate_python(df.to_dict(orient="records"))
        return pd.DataFrame([item.model_dump(exclude_none=True) for item in validated])