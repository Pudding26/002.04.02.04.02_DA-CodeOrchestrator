import logging
import json
import yaml
import pprint
import pandas as pd
import httpx

from app.tasks.TaskBase import TaskBase
from app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler





class TA35_0_BackendOrchestrator(TaskBase):
    def setup(self):
        logging.debug3("🔧 [TA35] Setup started.")
        self.src_HDF5_inst_1 = SWMR_HDF5Handler(self.instructions["src_db_path_1"])
        self.src_SQLiteHandler_inst_2 = self.instructions["src_SQLiteHandler"]
        self.dest_SQLiteHandler_inst_2 = self.instructions["dest_SQLiteHandler"]
        self.doe_df_raw = pd.DataFrame()
        self.ml_table_raw = pd.DataFrame()
        self.doe_df = pd.DataFrame()
        self.doe_job_list = []
        self.api_base_url = self.instructions.get("api_base_url", "http://localhost:8000")
        logging.debug3("✅ [TA35] Setup complete.")

    def run(self):
        try:
            self.controller.update_message("🔄 Starting DoE pipeline orchestration")

            self.trigger_task_via_http("TA31_0_DesignOfExperiments")

            if self.instructions.get("update_HDF5"):
                self.trigger_task_via_http("TA23_0_CreateWoodMaster")
                self.trigger_task_via_http("TA25_0_CreateWoodHDF")

            self.create_job_df()
            self.create_job_queue()

            self.trigger_task_via_http("TA30_B_SegmentationOrchestrator")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
        except Exception as e:
            logging.exception("❌ [TA35] Pipeline orchestration failed")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logging.debug3("🧼 [TA35] Cleanup complete.")

    def trigger_task_via_http(self, task_name):
        url = f"{self.api_base_url}/tasks/start"
        payload = {"task_name": task_name}

        logging.debug(f"🌐 Triggering subtask via HTTP POST: {url} with payload {payload}")
        try:
            res = httpx.post(url, json=payload)
            if res.status_code == 200:
                logging.info(f"✅ Successfully triggered {task_name}")
            else:
                logging.error(f"❌ Failed to trigger {task_name}: {res.status_code} {res.text}")
        except Exception as e:
            logging.exception(f"❌ HTTP error while triggering {task_name}: {e}")

    def create_job_df(self):
        def _deserialize(df):
            return df.applymap(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else x)

        logging.debug3("📥 Loading DoE and ML tables.")
        self.doe_df_raw = _deserialize(self._load_table(self.src_SQLiteHandler_inst_2, self.instructions["src_db_name_2"]))
        self.ml_table_raw = _deserialize(self._load_table(self.src_SQLiteHandler_inst_2, self.instructions["src_db_name_2B"]))

        if self.ml_table_raw.empty:
            self.doe_df = self.doe_df_raw
        else:
            self.doe_df = self.doe_df_raw[~self.doe_df_raw["DoE_UUID"].isin(self.ml_table_raw["DoE_UUID"])]

        logging.info(f"✅ Loaded {len(self.doe_df)} DoE jobs.")

    def create_job_queue(self):
        self.doe_job_list = []

        for idx, row in self.doe_df.iterrows():
            self.check_control()
            row_dict = dict(row)
            logging.debug3(f"🧩 Rendering job #{idx}: {row_dict}")
            job = self._render_template(row_dict)
            logging.debug3(f"✅ Job created: {job}")
            self.doe_job_list.append(job)

        self.controller.update_item_count(len(self.doe_job_list))
        logging.info(f"✅ Total jobs rendered: {len(self.doe_job_list)}")

    def _render_template(self, row_dict):
        with open("config/templates/DoE_job_template.yaml", "r") as f:
            template = yaml.safe_load(f)

        def fill(node):
            if isinstance(node, dict):
                return {k: fill(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [fill(v) for v in node]
            elif isinstance(node, str) and node.startswith("{") and node.endswith("}"):
                return row_dict.get(node[1:-1], None)
            return node

        job = fill(template)
        job["DoE_UUID"] = row_dict.get("DoE_UUID")
        return job

    def _load_table(self, handler, table_name):
        try:
            return handler.get_complete_Dataframe(table_name=table_name)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load table {table_name}: {e}")
            return pd.DataFrame()
