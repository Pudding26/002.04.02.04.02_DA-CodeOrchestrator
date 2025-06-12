import re
import os
import pandas as pd
import logging
if os.getenv("DEBUG_MODE") == "True":
    import memory_profiler
    memory_profiler.profile.disable = lambda: None
from app.tasks.TaskBase import TaskBase
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.controlling.TaskController import TaskController
from app.utils.mapping.YamlColumnMapper import YamlColumnMapper
from app.utils.SQL.SQLiteHandler import SQLiteHandler

class TA12_D_Transfer_DS11(TaskBase):
    def setup(self):
        logging.info("🔧 Setting up TA12_D_Transfer_DS11")
        self.controller.update_message("Initialized DS11 Transfer")
        self.controller.update_progress(0.01)

        logging.debug3(f"🧾 Instructions received: {self.instructions}")

    def run(self):
        try:
            logging.info("🚀 Run started for TA12_D_Transfer_DS11")
            self.controller.update_message("Step 1: Load raw data")
            df = self._load_raw_data()
            logging.debug3(f"📊 Loaded raw data: shape={df.shape}, columns={list(df.columns)}")
            self.check_control()
            self.controller.update_progress(0.2)

            self.controller.update_message("Step 2: Rename columns")
            df = YamlColumnMapper.rename_columns(
                df,
                self.instructions["path_gen_col_rename_mapper"],
                keys_list=["col_names"]
            )
            logging.debug3(f"🪄 Columns after renaming: {list(df.columns)}")
            self.check_control()
            self.controller.update_progress(0.35)

            self.controller.update_message("Step 3: Format taxonomy")
            df = self._preprocess_species(df)
            logging.debug3("🧬 Taxonomy preprocessing complete")
            self.check_control()
            self.controller.update_progress(0.45)

            self.controller.update_message("Step 4: Inject static metadata")
            df = YamlColumnMapper.add_static_columns(df, self.instructions["path_gen_manual_col_mapper"], ["TA12_D_Transfer_DS11"])
            logging.debug3("📥 Static column injection complete")
            self.check_control()
            self.controller.update_progress(0.6)

            self.controller.update_message("Step 5: Apply value mapping")
            df = YamlColumnMapper.yaml_col_value_mapper(self.instructions["path_value_mapper"], "DS11", df)
            logging.debug3("🧾 Value mapping complete")
            self.check_control()
            self.controller.update_progress(0.75)

            self.controller.update_message("Step 6: Assign specimen and shot numbers")
            df = self._assign_specimen_and_shot(df)
            logging.debug3("🔢 Specimen and shot number assignment complete")
            self.check_control()
            self.controller.update_progress(0.9)

            self.controller.update_message("Step 7: Store final data to SQL")
            self._store_data(df)
            logging.info(f"💾 Stored data to table: {self.instructions['dest_table_name']}")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            logging.info("✅ Task completed successfully")

        except Exception as e:
            logging.exception("❌ Task failed:")
            self.controller.finalize_failure(str(e))
            raise

        finally:
            self.cleanup()

    def cleanup(self):
        logging.debug5("🧹 Running cleanup")
        self.set_needs_running(False) #mark as already processed for the wrapper
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logging.debug5("📦 Cleanup and archival complete")

    def _load_raw_data(self):
        logging.debug3(f"📂 Loading SQLite DB from: {self.instructions['path_DataRaw']}")
        sqlite = SQLiteHandler(
            db_name=self.instructions["path_DataRaw"],
            can_read=True,
            can_write=False
        )
        df = sqlite.get_complete_Dataframe(self.instructions["src_table_name"])
        sqlite.close_connection()
        return df

    def _store_data(self, df):
        logging.debug3("🧼 Normalizing data types before storage")
        df = df.astype(object).where(pd.notnull(df), None)
        df = df.map(lambda x: x.item() if hasattr(x, 'item') else x)

        sql = SQL_Df(db_key=self.instructions["dest_db_name"])
        sql.store(self.instructions["dest_table_name"], df, method="replace")

    def _preprocess_species(self, df):
        logging.debug5("🔠 Cleaning taxonomy fields: species, genus, family")
        df["species"] = df["species"].apply(lambda x: ''.join([w.capitalize() for w in x.split(" ")]))
        df["genus"] = df["species"].apply(lambda x: re.findall(r'[A-Z][a-z]*', x)[0])
        df["family"] = df["family"].apply(lambda x: ''.join(w.capitalize() for w in x.split()))
        return df

    def _assign_specimen_and_shot(self, df):
        df["shotNo"] = df.groupby(["species", "specimenNo_old"]).cumcount() + 1
        df["specimenNo"] = df.groupby("species")["specimenNo_old"].transform(lambda x: pd.factorize(x)[0] + 1)
        return df
