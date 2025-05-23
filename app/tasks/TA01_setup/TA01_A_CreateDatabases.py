import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()  # Load environment variables from .env


class TA01_A_CreateDatabases:
    def __init__(self, user, password, host, port, default_db="postgres"):
        self.connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{default_db}"
        self.engine = create_engine(self.connection_url, isolation_level="AUTOCOMMIT")

    def database_exists(self, db_name):
        query = text("SELECT 1 FROM pg_database WHERE datname = :db")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"db": db_name})
            return result.scalar() is not None

    def create_database(self, db_name):
        if self.database_exists(db_name):
            print(f"Database '{db_name}' already exists.")
        else:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(f"CREATE DATABASE {db_name}"))
                    print(f"Database '{db_name}' created.")
            except SQLAlchemyError as e:
                print(f"Failed to create database '{db_name}': {e}")

    def create_multiple(self, db_list):
        for db_name in db_list:
            self.create_database(db_name)


def run_classname():
    db_init = DatabaseInitializer(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    db_init.create_multiple(["source", "raw", "production"])


if __name__ == "__main__":
    run_classname()
