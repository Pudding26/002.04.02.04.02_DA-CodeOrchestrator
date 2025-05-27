import logging
import pandas as pd

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.models.production.api.api_WoodTableA import WoodTableAOut
from app.utils.SQL.models.production.api.api_DS09 import DS09Out
from app.utils.SQL.models.raw.api.api_primaryDataRaw import PrimaryDataRawOut


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

            records = [WoodTableAOut(**row._asdict()) for row in final_df.itertuples(index=False)]
            WoodTableAOut.persist_to_db(records)
            logging.info(f"✅ Stored {len(records)} records into WoodTableA ORM")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()

        except Exception as e:
            logging.error(f"❌ Task failed: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def load_needed_data(self):
        data_dict = {}

        logging.debug2("Fetching raw primary data from ORM...")
        raw_data = PrimaryDataRawOut.fetch_all()
        raw_df = pd.DataFrame([r.model_dump() for r in raw_data])
        data_dict["Data1Raw"] = raw_df

        logging.debug2("Fetching DS09 data from ORM...")
        ds09_data = DS09Out.fetch_all()
        ds09_df = pd.DataFrame([r.model_dump() for r in ds09_data])
        data_dict["DS09"] = ds09_df if not ds09_df.empty else None

        return data_dict

    def clean_and_enrich_data(self, data_dict):
        cols = ["family", "IFAW_ID", "genus", "species", "engName", "deName", "frName", "japName", "origin"]
        df_with_fam = pd.DataFrame()

        for name, df in data_dict.items():
            df = df.loc[:, df.columns.intersection(cols + ["genus", "family"])]
            data_dict[name] = df
            if "family" in df.columns:
                fam = df.dropna(subset=["family"])[["family", "genus"]]
                df_with_fam = pd.concat([df_with_fam, fam])

        fam_reference = df_with_fam.drop_duplicates()
        df_raw = data_dict["Data1Raw"]

        with_fam = df_raw[df_raw["family"].notna()]
        no_fam = df_raw[df_raw["family"].isna()].drop(columns="family")
        enriched = no_fam.merge(fam_reference, on="genus", how="left").dropna(subset=["family"])

        return pd.concat([with_fam, enriched])

    def merge_with_DS09(self, merged, DS09):
        if DS09 is None:
            logging.warning("⚠️ DS09 reference table missing. Skipping merge.")
            return merged

        DS09 = DS09.dropna(subset=["species"]).copy()
        DS09["keep_drop"] = True

        merged = merged.merge(DS09, on="species", how="left", suffixes=('', '_DS09'))
        for col in [c for c in DS09.columns if c != "species"]:
            if f"{col}_DS09" in merged.columns:
                merged[col] = merged[f"{col}_DS09"].combine_first(merged[col])
                merged.drop(columns=[f"{col}_DS09"], inplace=True)

        return merged

    def filter_data(self, df):
        df['keep_drop'] = df['sourceNo'].isin(["DS01", "DS04", "DS07"])
        df = df[df['keep_drop']].drop(columns=["keep_drop"])

        drop_cols = [c for c in df.columns if c.endswith(("drop", "temp", "old"))]
        return df.drop(columns=drop_cols)
