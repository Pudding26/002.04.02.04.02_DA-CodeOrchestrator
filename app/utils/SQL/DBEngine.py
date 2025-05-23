# app/db/base.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

class DBEngine:
    def __init__(self, db_key: str):
        """
        db_key: The environment key like 'source_db', 'raw_db', etc.
        """
        self.db_key = db_key
        self.database_url = os.getenv(db_key)

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
