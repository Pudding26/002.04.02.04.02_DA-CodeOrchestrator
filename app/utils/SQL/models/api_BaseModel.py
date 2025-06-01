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
        Validate DataFrame rows using Pydantic, then persist via SQLAlchemy ORM.
        Supported methods: 'append' (default), 'replace' (drops and inserts).
        """
        from app.utils.SQL.DBEngine import DBEngine
        from sqlalchemy.orm import sessionmaker

        engine = DBEngine(db_key).get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

        # Optional: Replace all existing rows
        if method == "replace":
            logging.warning(f"⚠️ 'replace' will DELETE all existing rows in {cls.orm_class.__tablename__}")
            try:
                session.query(cls.orm_class).delete()
                session.commit()
            except Exception as e:
                session.rollback()
                logging.error(f"❌ Failed to delete existing data from {cls.orm_class.__tablename__}: {e}", exc_info=True)
                raise

        # Validate with Pydantic
        try:
            validated_records = cls._model_validate_dataframe(df)
        except Exception as e:
            logging.error(f"❌ Pydantic validation failed in {cls.__name__}: {e}", exc_info=True)
            raise

        # Convert to ORM instances
        try:
            orm_objects = [cls.orm_class(**record) for record in validated_records.to_dict(orient="records")]
            session.bulk_save_objects(orm_objects)
            session.commit()
            logging.info(f"✅ Stored {len(orm_objects)} rows to {cls.orm_class.__tablename__} using ORM ({method})")
        except Exception as e:
            session.rollback()
            logging.error(f"❌ Failed to store ORM records in {cls.__name__}: {e}", exc_info=True)
            raise
        finally:
            session.close()





    @classmethod
    def fetch_all(cls: Type[T], db_key: str = None, orm_class: Optional[Type] = None) -> pd.DataFrame:
        """
        Fetch all entries from the database and return as a DataFrame.
        If the query succeeds but returns no rows, return a DataFrame with correct schema.
        If the query fails, return a truly empty DataFrame.
        """
        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")

        if orm_class is None:
            raise ValueError(f"{cls.__name__} must define 'orm_class' or pass it explicitly.")
        
        session = DBEngine(db_key).get_session()
        logging.debug2(f"🔍 Fetching all entries from {orm_class.__name__} in {db_key} database")

        try:
            results = session.query(orm_class).all()

            if not results:
                logging.debug1(f"📭 No entries found in table {orm_class.__tablename__}")
                field_names = list(cls.model_fields)
                return pd.DataFrame(columns=field_names)

            validated = [cls.model_validate(row).model_dump() for row in results]
            return pd.DataFrame(validated)

        except Exception as e:
            logging.error(f"❌ Failed to fetch {orm_class.__name__} entries: {e}", exc_info=True)
            return pd.DataFrame()
        finally:
            session.close()



    @classmethod
    def _model_validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        adapter = TypeAdapter(list[cls])
        validated = adapter.validate_python(df.to_dict(orient="records"))
        return pd.DataFrame([item.model_dump(exclude_none=True) for item in validated])
    

    @classmethod
    def validate_df(cls, df: pd.DataFrame) -> List[dict]:
        """
        Validate DataFrame rows using Pydantic. Returns a list of validation errors, if any.
        Each error is a dict: {'row': index, 'error': str, 'row_data': dict}
        """
        errors = []
        records = df.to_dict(orient="records")

        for idx, record in enumerate(records):
            try:
                cls.model_validate(record)
            except Exception as e:
                errors.append({
                    "row": idx,
                    "error": str(e),
                    "row_data": record
                })
        return errors


    @classmethod
    def db_shape(cls, db_key: str = None, orm_class: Optional[Type] = None) -> tuple[int, int]:
        """
        Return the shape (rows, columns) of the SQL table mapped to this model.
        """
        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")

        if orm_class is None:
            raise ValueError(f"{cls.__name__} must define 'orm_class' or pass it explicitly.")

        session = DBEngine(db_key).get_session()

        try:
            row_count = session.query(orm_class).count()
            column_count = len(cls.model_fields)
            return (row_count, column_count)
        except Exception as e:
            logging.error(f"❌ Failed to get DB shape for {orm_class.__tablename__}: {e}", exc_info=True)
            return (0, 0)
        finally:
            session.close()
