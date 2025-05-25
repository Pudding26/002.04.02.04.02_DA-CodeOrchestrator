# ORM Base with shared method for DataFrame persistence

from sqlalchemy.orm import declarative_base, Session
import pandas as pd
import logging


class _Base:
    @classmethod
    def store_dataframe(cls, df: pd.DataFrame, db_key: str, method: str = "append"):
        """
        Store a DataFrame into the SQL table represented by this ORM model.

        Parameters:
        - df: pd.DataFrame
            The DataFrame to store. Column names must match the ORM model fields.
        - db_key: str
            The name of the database (as registered in DBEngine, e.g., 'production', 'raw').
        - method: str
            Either 'append' (default) or 'replace'.
            'append' adds new rows.
            'replace' deletes all existing rows before inserting.

        Example:
            DS09.store_dataframe(df, db_key="production", method="replace")
        """
        from app.utils.SQL.DBEngine import DBEngine
        session: Session = DBEngine(db_key).get_session()
        try:
            with session.begin():  # begin a transaction block
                if method == "replace":
                    session.query(cls).delete()
                    session.flush()  # ← ensures deletion is executed before insert
                    logging.debug2(f"🧹 Cleared existing rows from {cls.__tablename__}")

                objs = [cls(**row.dropna().to_dict()) for _, row in df.iterrows()]
                session.bulk_save_objects(objs)
                logging.info(f"✅ Stored {len(objs)} rows to {cls.__tablename__} using ORM ({method})")

        except Exception as e:
            logging.error(f"❌ Failed to store to {cls.__tablename__}: {e}", exc_info=True)
            raise

        finally:
            session.close()


# Base class for all ORM models
OrmBase = declarative_base(cls=_Base)
