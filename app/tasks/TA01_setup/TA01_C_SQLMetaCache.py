import os
import yaml
import logging
from pathlib import Path
from sqlalchemy import inspect, text
from app.utils.SQL.DBEngine import DBEngine
from app.tasks.TaskBase import TaskBase

class TA01_C_SQLMetaCache(TaskBase):
    """
    Task that scans all databases and caches filter schema YAML files.
    Includes reusable static methods for single-table update and multi-table read.
    """

    CACHE_BASE = "app/cache/SQL"

    def setup(self):
        logging.info("⚙️ Setting up SQLMetaCache task...")
        self.db_keys = ["raw", "source", "production", "progress", "jobs"]

    def run(self):
        logging.info("🚀 Running SQLMetaCache task...")
        total_cached = 0
        for db_key in self.db_keys:
            try:
                db = DBEngine(db_key)
                engine = db.get_engine()
                insp = inspect(engine)
                tables = insp.get_table_names()
                for table in tables:
                    self.check_control()
                    if self.update_cache(db_key, table):
                        total_cached += 1
            except Exception as e:
                logging.warning(f"⚠️ Could not process DB '{db_key}': {e}")
        
        logging.info(f"🎉 SQLMetaCache task complete: {total_cached} tables cached.")

    def cleanup(self):
        logging.info("🧹 Cleaning up SQLMetaCache task.")

    @staticmethod
    def update_cache(db_key: str, table_name: str):
        """
        Standalone method to update cache for a single db_key.table_name
        """
        try:
            db = DBEngine(db_key)
            engine = db.get_engine()
            insp = inspect(engine)
            columns_info = insp.get_columns(table_name)
            metadata = []

            with engine.connect() as conn:
                for col in columns_info:
                    col_name = col["name"]
                    col_type = str(col["type"]).lower()
                    col_meta = {"name": col_name}

                    if any(kw in col_type for kw in ["int", "float", "numeric", "real", "double"]):
                        result = conn.execute(text(f'SELECT MIN(\"{col_name}\"), MAX(\"{col_name}\") FROM \"{table_name}\"'))
                        min_val, max_val = result.fetchone()
                        col_meta.update({
                            "type": "numeric",
                            "min": min_val,
                            "max": max_val
                        })
                    else:
                        result = conn.execute(text(f'SELECT COUNT(DISTINCT \"{col_name}\") FROM \"{table_name}\"'))
                        unique_count = result.scalar()
                        col_meta["unique_count"] = unique_count
                        col_meta["type"] = "categorical"

                        if unique_count <= 10:
                            result = conn.execute(text(f'SELECT DISTINCT \"{col_name}\" FROM \"{table_name}\" LIMIT 10'))
                            unique_vals = [row[0] for row in result.fetchall()]
                            col_meta["unique_values"] = unique_vals

                    metadata.append(col_meta)

            out_dir = os.path.join(TA01_C_SQLMetaCache.CACHE_BASE, db_key)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{table_name}.yaml")

            with open(out_path, "w") as f:
                yaml.dump({"columns": metadata}, f)

            logging.debug1(f"✅ Updated filter schema cache: {db_key}.{table_name} -> {out_path}")
            return True

        except Exception as e:
            logging.warning(f"⚠️ Failed to update cache for {db_key}.{table_name}: {e}")
            return False

    @staticmethod
    def get_all_table_caches():
        """
        Standalone method to load all individual YAML caches into one large dict:
        {
          db_key: {
            table_name: { "columns": [...] }
          }
        }
        """
        all_cache = {}
        base = TA01_C_SQLMetaCache.CACHE_BASE

        if not os.path.exists(base):
            logging.info(f"ℹ️ No cache directory found at {base}")
            return all_cache

        for db_key in os.listdir(base):
            db_path = os.path.join(base, db_key)
            if not os.path.isdir(db_path):
                continue

            all_cache[db_key] = {}

            for file in os.listdir(db_path):
                if file.endswith(".yaml"):
                    table_name = file[:-5]  # strip .yaml
                    file_path = os.path.join(db_path, file)
                    try:
                        with open(file_path, "r") as f:
                            data = yaml.safe_load(f)
                            all_cache[db_key][table_name] = data
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to load cache {file_path}: {e}")

        return all_cache
