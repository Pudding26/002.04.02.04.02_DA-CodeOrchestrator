import json, yaml
import pandas as pd
import logging

from app.tasks.TaskBase import TaskBase
from app.tasks.TA27_DesignOfExperiments.TA27_B_DoEExpander import TA27_B_DoEExpander
from app.tasks.TA27_DesignOfExperiments.TA27_A_DoEJobGenerator import TA27_A_DoEJobGenerator
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.YAML.YAMLUtils import YAMLUtils

from app.utils.SQL.models.production.api.api_DoEArchive import DoEArchive_Out
from app.utils.SQL.models.temp.api.api_DoEJobs import DoEJobs_Out



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
            logger.debug3(f"🔢 Expanded DoE to {len(df)} combinations")

            #df_safe = df.map(lambda x: json.dumps(x) if isinstance(x, list) else x)
            df_safe = df
            self.controller.update_message("📦 Archiving old DoE")
            logging.debug3("📦 Archiving old DoE table if exists...")
            self.archive_old_doe(df_safe)
            logger.debug3("📦 Old DoE archived successfully")
            
            self.controller.update_message("🧪 Saving new DoEJobs to SQL")
            logger.debug3("💾 Storing new DoEJobs in SQL...")
            DoEJobs_Out.store_dataframe(df_safe, method="replace", db_key="temp")
            
            logger.info("✅ New DoE table stored")

            self.controller.update_message("🛠 Generating job definitions")
            logger.debug3("🧬 Starting job generation...")
            jobs = TA27_A_DoEJobGenerator.generate(df, self.instructions["job_template_path"])
            logger.debug3(f"📦 Generated {len(jobs)} job definitions")
            
            return jobs
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
        df_old = DoEJobs_Out.fetch_all()
        logging.debug2(f"📂 Found {len(df_old)} old DoE rows to archive")
        df_archive_raw = DoEArchive_Out.fetch_all()
        logging.debug2(f"📂 Found {len(df_archive_raw)} DoE rows in Archive")
        new_rows = df_old[~df_old['DoE_UUID'].isin(df_archive_raw['DoE_UUID'])]
        
        DoEArchive_Out.store_dataframe(new_rows, method="append", db_key="production")
        logger.debug3(f"📂 Archived {len(new_rows)} rows to backup table")

    def cleanup(self):
        logger.debug3("🧹 Running cleanup phase...")
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logger.debug3("🧼 Cleanup complete.")
