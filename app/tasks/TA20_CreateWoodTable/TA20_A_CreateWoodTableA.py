import logging
import pandas as pd

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.SQL.models.production.api.api_DS09 import DS09_Out
from app.utils.SQL.models.raw.api.api_primaryDataRaw import PrimaryDataRaw_Out


class TA20_A_CreateWoodTableA(TaskBase):
    def setup(self):
        self.controller.update_message("Initialized wood table construction")
        logging.debug2("✅ Task setup complete.")

    def run(self):
        try:
            self.controller.update_message("Loading source data")
            data_dict = self.load_needed_data()
            logging.debug2(f"📦 Loaded {len(data_dict)} source tables")

            cleaned_df = self.clean_and_enrich_data(data_dict)
            logging.debug2("🧹 Cleaned and enriched source data")

            merged_df = self.merge_with_DS09(cleaned_df, data_dict.get("DS09"))
            logging.debug2("🔗 Merged with DS09 reference")

            final_df = self.filter_data(merged_df)
            logging.debug2(f"🔍 Filtered final dataset: {len(final_df)} rows")
            final_df = final_df.where(pd.notna(final_df), None)
            logging.debug2("📦 Final DataFrame ready for storage")

            WoodTableA_Out.store_dataframe(final_df, db_key="production", method="replace")

            logging.info(f"✅ Stored {len(final_df)} records into WoodTableA ORM")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()

        except Exception as e:
            logging.error(f"❌ Task failed: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logging.debug2("🧹 Starting cleanup...")
        self.flush_memory_logs()
        logging.debug2("📦 Archiving logs with ORM controller...")
        self.controller.archive_with_orm()
        logging.debug2("✅ Cleanup complete.")

    def load_needed_data(self):
        data_dict = {}

        logging.debug2("📥 Fetching raw primary data from ORM...")
        raw_data = PrimaryDataRaw_Out.fetch_all()
        raw_df = pd.DataFrame([r.model_dump() for r in raw_data])
        data_dict["Data1Raw"] = raw_df
        logging.debug2(f"✅ Fetched {len(raw_df)} rows of raw primary data.")

        logging.debug2("📥 Fetching DS09 reference data from ORM...")
        ds09_data = DS09_Out.fetch_all()
        ds09_df = pd.DataFrame([r.model_dump() for r in ds09_data])
        data_dict["DS09"] = ds09_df if not ds09_df.empty else None
        logging.debug2(f"✅ Fetched {len(ds09_df)} rows from DS09." if ds09_df is not None else "⚠️ DS09 data is empty.")

        return data_dict

    def clean_and_enrich_data(self, data_dict):
        logging.debug2("🧽 Starting cleaning and enrichment of raw data...")
        cols = ["sourceNo", "family", "IFAW_code", "genus", "species", "engName", "deName", "frName", "japName", "origin"]
        df_with_fam = pd.DataFrame()

        for name, df in data_dict.items():
            if df is None:
                logging.debug2(f"⚠️ Skipping '{name}' as it is None.")
                continue
            logging.debug2(f"🔍 Processing columns for dataset '{name}'...")
            df = df.loc[:, df.columns.intersection(cols + ["genus", "family"])]
            data_dict[name] = df
            if "family" in df.columns:
                fam = df.dropna(subset=["family"])[["family", "genus"]]
                df_with_fam = pd.concat([df_with_fam, fam])
                logging.debug2(f"📊 Found {len(fam)} non-null family entries in '{name}'.")

        fam_reference = df_with_fam.drop_duplicates()
        logging.debug2(f"🧬 Constructed family reference table with {len(fam_reference)} unique rows.")

        df_raw = data_dict["Data1Raw"]

        with_fam = df_raw[df_raw["family"].notna()]
        no_fam = df_raw[df_raw["family"].isna()].drop(columns="family")
        enriched = no_fam.merge(fam_reference, on="genus", how="left").dropna(subset=["family"])

        logging.debug2(f"🔗 Enriched {len(enriched)} rows by merging missing family values.")
        logging.debug2(f"📦 Final raw data size after enrichment: {len(with_fam) + len(enriched)} rows.")

        return pd.concat([with_fam, enriched])

    def merge_with_DS09(self, merged, DS09):
        logging.debug2("🔗 Merging cleaned data with DS09 reference table...")
        if DS09 is None:
            logging.warning("⚠️ DS09 reference table missing. Skipping merge.")
            return merged

        DS09 = DS09.dropna(subset=["species"]).copy()
        logging.debug2(f"🔍 Merging on {len(DS09)} non-null species rows.")

        DS09["keep_drop"] = True
        merged = merged.merge(DS09, on="species", how="left", suffixes=('', '_DS09'))

        updated_cols = 0
        for col in [c for c in DS09.columns if c != "species"]:
            ds09_col = f"{col}_DS09"
            if ds09_col in merged.columns:
                merged[col] = merged[ds09_col].combine_first(merged[col])
                merged.drop(columns=[ds09_col], inplace=True)
                updated_cols += 1

        logging.debug2(f"✅ Merged and updated {updated_cols} columns from DS09.")
        return merged

    def filter_data(self, df):
        logging.debug2("🚿 Filtering merged data by source and dropping temp columns...")
        original_len = len(df)

        df['keep_drop'] = df['sourceNo'].isin(["DS01", "DS04", "DS07"])
        df = df[df['keep_drop']].drop(columns=["keep_drop"])

        drop_cols = [c for c in df.columns if c.endswith(("drop", "temp", "old"))]
        df = df.drop(columns=drop_cols)

        logging.debug2(f"✅ Filtered data: {original_len} → {len(df)} rows. Dropped columns: {drop_cols}")
        return df
