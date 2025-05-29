# app/db/base.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy.schema import CreateTable

import logging


load_dotenv()

def create_all_tables():
    
    #progress
    from app.utils.SQL.models.progress.orm.ProfileArchive import ProfileArchive
    from app.utils.SQL.models.progress.orm.ProgressArchive import ProgressArchive
    
    # raw
    from app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw
    
    # production
    from app.utils.SQL.models.production.orm.DS09 import DS09
    from app.utils.SQL.models.production.orm.DS40 import DS40
    from app.utils.SQL.models.production.orm.WoodTableA import WoodTableA

    grouped_models = {
        "progress": [ProfileArchive, ProgressArchive],
        "raw": [PrimaryDataRaw],
        "production": [WoodTableA, DS09, DS40],
    }

    for db_key, model_list in grouped_models.items():
        try:
            db = DBEngine(db_key)
            engine = db.get_engine()
            for model in model_list:
                model.__table__.create(bind=engine, checkfirst=True)
                logging.debug3(f"✅ Created table '{model.__tablename__}' in {db_key}")

        except Exception as e:
            logging.debug3(f"❌ Failed to create tables for {db_key}: {e}")





class DBEngine:
    def __init__(self, db_key: str):
        """
        db_key: The environment key like 'source_db', 'raw_db', etc.
        """
        self.db_key = db_key
        self.database_url = self._build_pg_url(db_key)
        
        if not self.database_url:
            raise ValueError(f"❌ No DB URL found in .env for key: {db_key}")

        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self):
        """Creates a new SQLAlchemy session bound to the engine."""
        return self.SessionLocal()

    def get_engine(self):
        """Returns the raw SQLAlchemy engine (for pandas or raw SQL)."""
        return self.engine

    def test_connection(self):
        """Simple test query to confirm DB connection is working."""
        with self.get_engine().connect() as conn:
            conn.execute("SELECT 1")

    def _build_pg_url(self, db_key: str) -> str:
        user = os.getenv("DB_USER")
        pwd = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db = os.getenv(f"DB_{db_key.upper()}_NAME")  # e.g. DB_SOURCE_NAME

        if not all([user, pwd, host, port, db]):
            raise ValueError(f"❌ Missing env vars for {db_key} DB")

        return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"