import os
import logging
import json
import yaml
import pandas as pd
from memory_profiler import profile
from datetime import datetime
from typing import Optional
import numpy as np
from uuid import uuid4

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out
from app.utils.SQL.models.temp.api.api_PrimaryDataJobs import PrimaryDataJobs_Out
from app.utils.HDF5.HDF5_Inspector import HDF5Inspector



WOOD_MASTER_AGG_CONFIG = {
    "group_first_cols": [
        "woodType", "family", "genus", "species",
        "engName", "deName", "frName", "japName",
        "sourceID", "sourceNo", "specimenID",
        "microscopicTechnic", "institution", "contributor", "digitizedDate",
        "view", "lens", "totalNumberShots", "pixelSize_um_per_pixel",
        "DPI", "hdf5_dataset_path", "samplingPoint", "origin", "institutionCode", "citeKey",
        "numericalAperature_NA", "area_x_mm", "area_y_mm", "IFAW_code", "raw_UUID",
        "GPS_Alt", "GPS_Lat", "GPS_Long"
    ],
    "group_list_cols": ["sourceFilePath_rel"]
}

later_cols = ["filterNo", "colorDepth", "colorSpace", "pixel_x", "pixel_y", "bitDepth"]

class TA23_0_CreateWoodMaster(TaskBase):
    def setup(self):
        self.df_writer = SQL_Df(self.instructions["Thread_progress_db_path"])
        logging.info("Setup complete. SQL writer initialized.")
        self.controller.update_message("Setup complete.")

    def run(self):
        try:
            self.controller.update_message("Loading woodTable...")
            logging.info("🔄 Loading data from WoodTableA_Out...")
            raw_df = WoodTableA_Out.fetch_all()
            logging.debug2(f"Loaded {len(raw_df)} rows from woodTable.")

            self.controller.update_message("Cleaning woodTable...")
            wood_df = self.clean_woodTable(raw_df)
            logging.debug2(f"Cleaned woodTable: {wood_df.shape[0]} rows, {wood_df.shape[1]} columns.")

            self.controller.update_message("Creating new woodMaster...")
            wood_new = self.create_woodMaster_new(wood_df)
            logging.debug2(f"New woodMaster created: {wood_new.shape}")

            self.controller.update_message("Refreshing HDF5 woodMaster...")
            TA23_0_CreateWoodMaster.refresh_woodMaster(self.instructions["HDF5_file_path"])

            self.controller.update_message("Loading old woodMaster...")
            woodMaster_old = WoodMaster_Out.fetch_all()
            logging.debug2(f"Old woodMaster loaded: {len(woodMaster_old)} rows.")

            self.controller.update_message("Identifying new sampleIDs...")
            job_df = wood_new[~wood_new["sampleID"].isin(woodMaster_old["sampleID"])]
            logging.info(f"Identified {len(job_df)} new samples.")

            self.controller.update_message("Storing new jobs...")
            #job_df["sourceFilePath_rel"] = job_df["sourceFilePath_rel"].apply(json.dumps) #Not needed anymore, as SQL alchemycan handle lists. was needed for SQLite
            job_df = self.prepare_for_sql(job_df)

            logging.debug3(f"Prepared job DataFrame for SQL. Shape: {job_df.shape}")
            PrimaryDataJobs_Out.store_dataframe(job_df, db_key="temp", method="replace")
            logging.info("✅ Stored new job samples in temp DB.")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
        except Exception as e:
            logging.error(f"❌ Task failed: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()


    def cleanup(self):
        logging.info("🧹 Running cleanup and archiving task state.")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def clean_woodTable(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug3("🔧 Cleaning woodTable data...")

        def sanitize(val):
            return str(val).replace(" ", "_").replace("/", "-")

        def derive_ids(df):
            df["sourceID"] = df["species"] + "_" + df["sourceNo"]
            df["specimenID"] = df["sourceID"] + "_No" + df["specimenNo"].astype(int).astype(str).str.zfill(3)
            df["sampleID"] = df["specimenID"] + "_" + df["view"]
            return df

        def add_wood_type(df):
            with open(self.instructions["woodTypeMapper_path"], 'r') as f:
                wood_map = yaml.safe_load(f)
            df["woodType"] = df["family"].map(wood_map).fillna("unknown")
            return df

        def add_hdf5_path(df):
            df["hdf5_dataset_path"] = df.apply(lambda row: "/".join([
                sanitize(row["woodType"]), sanitize(row["family"]), sanitize(row["genus"]),
                sanitize(row["species"]), sanitize(row["sourceID"]),
                sanitize(row["specimenID"]), sanitize(row["sampleID"])
            ]), axis=1)
            return df

        df = derive_ids(df)
        df = add_wood_type(df)
        df = add_hdf5_path(df)
        logging.debug3("✅ woodTable cleaned.")
        return df

    def create_woodMaster_new(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug3("🧪 START: Creating the woodMaster_new.")
        logging.debug2(f"Initial woodTable shape: {df.shape}")

        agg_dict = {col: 'first' for col in WOOD_MASTER_AGG_CONFIG["group_first_cols"]}
        agg_dict.update({col: list for col in WOOD_MASTER_AGG_CONFIG["group_list_cols"]})

        result = df.groupby("sampleID", dropna=False).agg(agg_dict).reset_index()
        logging.debug3(f"Grouped woodTable: {result.shape[0]} rows")

        pydantic_order = list(PrimaryDataJobs_Out.model_fields.keys())
        reordered_cols = [col for col in pydantic_order if col in result.columns]
        other_cols = [col for col in result.columns if col not in reordered_cols]
        result = result[reordered_cols + other_cols]

        logging.debug2(f"Reordered woodMaster_new DataFrame: {result.shape[0]} rows, {result.shape[1]} columns")
        logging.debug3("✅ END: Created the woodMaster_new.")
        return result

    @staticmethod
    def refresh_woodMaster(hdf5_path: str) -> pd.DataFrame:
        logging.info(f"🔄 Refreshing woodMaster from: {hdf5_path}")

        if not os.path.exists(hdf5_path):
            logging.warning(f"❌ HDF5 file not found at path: {hdf5_path}")
            return pd.DataFrame(columns=list(PrimaryDataJobs_Out.model_fields.keys()))

        shape_old = WoodMaster_Out.db_shape()

        try:
            df = HDF5Inspector.HDF5_meta_to_df(hdf5_path)

            if df.empty:
                logging.warning("⚠️ HDF5 metadata DataFrame is empty.")
                return pd.DataFrame(columns=list(PrimaryDataJobs_Out.model_fields.keys()))

            if "dataset_shape_drop" in df.columns:
                df = df.drop(columns=["dataset_shape_drop"])

            df["stackID"] = df["path"].apply(lambda x: x.split("/")[-1])
            #df = TA23_0_CreateWoodMaster._reorder_woodMaster(df)

            WoodMaster_Out.store_dataframe(df, db_key="production", method="replace")
            shape_new = WoodMaster_Out.db_shape()

            logging.debug3(f"✅ Refreshed and stored woodMaster from HDF5. Old shape: {shape_old}, New shape: {shape_new}")
            return df

        except Exception as e:
            logging.error(f"❌ Failed to refresh woodMaster: {e}", exc_info=True)
            return pd.DataFrame(columns=list(PrimaryDataJobs_Out.model_fields.keys()))


    @staticmethod
    def _reorder_woodMaster(df: pd.DataFrame) -> pd.DataFrame:
        preferred = list(PrimaryDataJobs_Out.model_fields.keys())
        final_cols = preferred + [col for col in df.columns if col not in preferred]
        return df[final_cols]

    @staticmethod
    def prepare_for_sql(df: pd.DataFrame) -> pd.DataFrame:
        logging.debug3("🧪 Preparing DataFrame for SQL storage...")
        df = df.copy()
        
        def _clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            for col in df.columns:
                df[col] = df[col].map(lambda x: None if pd.isna(x) else x)
            return df

                
        
        
        orig_shape = df.shape

        df.replace(to_replace=["unknown", "null", "None", "[null]"], value=pd.NA, inplace=True)
        df["origin"] = df["origin"].replace("todo", "unknown")

        # Numeric coercion
        numeric_cols = [
            "DPI", "totalNumberShots", "area_x_mm", "area_y_mm",
            "numericalAperature_NA", "pixelSize_um_per_pixel",
            "GPS_Alt", "GPS_Lat", "GPS_Long", "lens"
        ]
        for col in numeric_cols:
            if col in df.columns:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Check if column can safely be cast to int (ignoring NaNs)
                col_series = df[col].dropna()
                is_integral = np.all(col_series == col_series.astype(int))

                if is_integral:
                    df[col] = df[col].astype(pd.Int64Dtype())  # Nullable integer type
                else:
                    df[col] = df[col].astype(float)

        if "digitizedDate" in df.columns:
            df["digitizedDate"] = pd.to_datetime(df["digitizedDate"], errors="coerce")
            df["digitizedDate"] = df["digitizedDate"].astype("object").where(df["digitizedDate"].notnull(), None) #pandas uses Nat by default for missing time values. sql alchemy cant handle


        for field in PrimaryDataJobs_Out.model_fields:
            if field not in df.columns:
                df[field] = pd.NA

        if "raw_UUID" in df.columns:
            df["raw_UUID"] = df["raw_UUID"].apply(
                lambda x: str(uuid4()) if pd.isna(x) or x in ["", "None", None] else x
            )

        df = df.where(pd.notnull(df), None)
        before_drop = len(df)
        df = df.dropna(subset=["sampleID"])
        dropped = before_drop - len(df)
        if dropped:
            logging.info(f"🗑️ Dropped {dropped} rows due to missing sampleID.")


        # Convert to proper int (Python int, not pandas nullable)
        int_columns = ['lens', 'totalNumberShots', 'DPI']
        for col in int_columns:
            df[col] = df[col].astype('float').astype('Int64')  # still nullable
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)

        contributor_mapper = {
            "DS01" : "Da Silva",
            "DS04" : "Junji Sugiyama",
            "DS07" : "J. Martins",
        }

        if "contributor" in df.columns and "sourceNo" in df.columns:
            df["contributor"] = df.apply(
                lambda row: contributor_mapper.get(row["sourceNo"], row["contributor"]),
                axis=1
    )

        df = _clean_nulls(df)

        logging.debug3(f"✅ Prepared DataFrame for SQL: {df.shape} (was {orig_shape})")
        return df
