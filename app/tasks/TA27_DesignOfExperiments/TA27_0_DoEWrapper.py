import json, yaml
import pandas as pd
import logging

from app.tasks.TaskBase import TaskBase
from app.tasks.TA27_DesignOfExperiments.TA27_B_DoEExpander import TA27_B_DoEExpander
from app.tasks.TA27_DesignOfExperiments.TA27_A_DoEJobGenerator import TA27_A_DoEJobGenerator
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.YAML.YAMLUtils import YAMLUtils


logger = logging.getLogger(__name__)

class TA27_0_DoEWrapper(TaskBase):

    def setup(self):
        logger.debug3("🔧 Setting up SQL interface...")
        self.sql = SQL_Df(self.instructions["dest_db_path_1"])
        self.controller.update_message("DoE Task Initialized.")

    def run(self):
        try:
            logger.info("🚀 Starting DoE task")
            self.controller.update_message("📂 Loading DoE YAML")
            logger.debug3(f"📥 Loading DoE YAML from: {self.instructions['doe_yaml_path']}")
            raw_yaml = YAMLUtils.load_yaml(self.instructions["doe_yaml_path"])

            logger.debug3("🧮 Expanding parameter combinations...")
            df = TA27_B_DoEExpander.expand(raw_yaml)
            df["DoE_UUID"] = df.apply(TA27_B_DoEExpander.short_hash, axis=1)
            logger.debug3(f"🔢 Expanded DoE to {len(df)} combinations")

            df_safe = df.applymap(lambda x: json.dumps(x) if isinstance(x, list) else x)

            self.controller.update_message("📦 Archiving old DoE")
            logger.info("📦 Archiving old DoE table if exists...")
            #self.archive_old_doe(df_safe)

            self.controller.update_message("🧪 Saving new DoE to SQL")
            logger.debug3(f"💾 Writing to table: {self.instructions['dest_db_table_name']}")
            #self.sql.store(self.instructions["dest_db_table_name"], df_safe, method="replace")
            logger.info("✅ New DoE table stored")

            self.controller.update_message("🛠 Generating job definitions")
            logger.debug3("🧬 Starting job generation...")
            jobs = TA27_A_DoEJobGenerator.generate(df, self.instructions["job_template_path"])
            logger.debug3(f"📦 Generated {len(jobs)} job definitions")

            output_yaml_path = self.instructions["output_jobs_yaml"]
            with open(output_yaml_path, "w") as f:
                yaml.dump({"jobs": jobs}, f, allow_unicode=True)
            logger.info(f"📝 Jobs written to YAML: {output_yaml_path}")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            logger.info("🎉 DoE task completed successfully.")

        except Exception as e:
            logger.error(f"❌ Error during DoE task: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise

    def archive_old_doe(self, new_df: pd.DataFrame):
        table_name = self.instructions["dest_db_table_name"]
        backup = f"{table_name}_backup"
        if table_name in self.sql.get_table_names():
            logger.info(f"📦 Archiving existing DoE table to: {backup}")
            df_old = self.sql.load(table_name).drop_duplicates(subset="DoE_UUID")
            self.sql.store(backup, df_old, method="replace")
            logger.debug3(f"📂 Archived {len(df_old)} rows to backup table")

    def cleanup(self):
        logger.debug3("🧹 Running cleanup phase...")
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logger.debug3("🧼 Cleanup complete.")
