from typing import Union, List, Optional
import numpy as np
import pandas as pd
import h5py
import logging
import os

from sqlalchemy import text
from sqlalchemy.orm import Session
from app.utils.SQL.DBEngine import DBEngine


from app.utils.SQL.models.production.orm.WoodMaster import WoodMaster
from app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out

class HDF5Inspector:

    @staticmethod
    def get_all_dataset_paths(file_path: str) -> list:
        """Return all dataset paths in the given HDF5 file."""
        paths = []

        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                paths.append(name)

        with h5py.File(file_path, 'r', swmr=True) as f:
            f.visititems(visitor)
        return paths
    

    @staticmethod
    def collect_attributes_for_dataset(file: h5py.File, dataset_path: str) -> dict:
        """
        Collect attributes from dataset path upward, assuming file is already open with swmr=True.
        """
        attrs = {}
        try:
            obj = file[dataset_path]
            attrs["dataset_shape_drop"] = obj.shape

            while obj.name != '/':
                for k, v in obj.attrs.items():
                    if k not in attrs:
                        attrs[k] = v
                obj = obj.parent

        except KeyError:
            logging.warning(f"Path {dataset_path} not found.")
        attrs["path"] = dataset_path
        return attrs

    @staticmethod
    def HDF5_meta_to_df(file_path: str) -> pd.DataFrame:
        """
        Retrieve a DataFrame of dataset paths and attributes using SWMR.
        """
        dataset_paths = HDF5Inspector.get_all_dataset_paths(file_path)
        with h5py.File(file_path, 'r', swmr=True) as f:
            records = [
                HDF5Inspector.collect_attributes_for_dataset(f, path)
                for path in dataset_paths
            ]
        return pd.DataFrame(records)

    @staticmethod
    def collect_attributes_for_dataset_threadSafe(file_path: str, dataset_path: str) -> dict:
        """
        Open file in SWMR mode for thread-safe and process-safe reads.
        """
        attrs = {}
        try:
            with h5py.File(file_path, 'r', swmr=True) as f:
                obj = f[dataset_path]
                attrs["dataset_shape_drop"] = obj.shape
                while obj.name != '/':
                    for k, v in obj.attrs.items():
                        if k not in attrs:
                            attrs[k] = v
                    obj = obj.parent
        except KeyError:
            logging.warning(f"Path {dataset_path} not found.")
        attrs["path"] = dataset_path
        return attrs



    @staticmethod
    def update_woodMaster_paths(
        hdf5_path: str,
        dataset_paths: Union[str, List[str]],
        db_key: str = "production",
        table_name: str = "woodMaster"
    ) -> Optional[pd.DataFrame]:
        """
        Update woodMaster using raw UPSERT, avoiding ORM. Uses session from DBEngine(db_key).
        """
        def _raw_upsert(df: pd.DataFrame, session: Session, table_name: str):
            if df.empty:
                logging.info("🟡 Skipping UPSERT — empty DataFrame.")
                return

            try:
                columns_raw = list(df.columns)                                # e.g. ["GPS_Lat", "family", ...]
                columns_quoted = [f'"{col}"' for col in columns_raw]          # e.g. ['"GPS_Lat"', '"family"', ...]
                placeholders = ", ".join(f":{col}" for col in columns_raw)    # e.g. ":GPS_Lat, :family, ..."

                update_clause = ", ".join(
                    f'{quoted} = EXCLUDED.{quoted}'
                    for col, quoted in zip(columns_raw, columns_quoted)
                    if col != "stackID"
                )

                sql = text(f"""
                    INSERT INTO "{table_name}" ({', '.join(columns_quoted)})
                    VALUES ({placeholders})
                    ON CONFLICT ("stackID") DO UPDATE SET
                    {update_clause}
                """)

                for row in df.to_dict(orient="records"):
                    session.execute(sql, params=row)
                session.commit()
                logging.debug(f"✅ Successfully upserted {len(df)} rows into '{table_name}'")

            except Exception as e:
                session.rollback()
                logging.error(f"❌ Raw UPSERT failed: {e}", exc_info=True)
                raise


            
        if not os.path.exists(hdf5_path):
            logging.warning(f"❌ HDF5 file not found: {hdf5_path}")
            return None

        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        try:
            expected_cols = set(WoodMaster_Out.model_fields.keys())

            # 🔒 Only enforce these as required
            required_fields = {
                "stackID", "sampleID", "woodType", "species", "family", "genus", "view",
                "lens", "totalNumberShots", "filterNo", "bitDepth", "colorDepth", "colorSpace",
                "pixel_x", "pixel_y", "citeKey", "sourceNo", "raw_UUID", "path",
                "stackID", "specimenID", "sourceID", "was_cropped"
            }

            records = []

            for path in dataset_paths:
                record = HDF5Inspector.collect_attributes_for_dataset_threadSafe(hdf5_path, path)
                if not record:
                    continue

                record["stackID"] = path.split("/")[-1]

                missing_required = required_fields - set(record.keys())
                if missing_required:
                    logging.warning(f"⚠️ Skipping path '{path}' — missing required fields: {missing_required}")
                    continue

                # Fill in missing optional fields as None
                for col in expected_cols:
                    if col not in record:
                        record[col] = None

                records.append(record)

            if not records:
                logging.info("ℹ️ No valid records to insert.")
                return None

            df = pd.DataFrame(records)

            if "dataset_shape_drop" in df.columns:
                df.drop(columns=["dataset_shape_drop"], inplace=True)

            df = df[[col for col in expected_cols if col in df.columns]]

            session: Session = DBEngine(db_key).get_session()
            _raw_upsert(df, session, table_name)

            logging.debug1(f"✅ Upserted {len(df)} rows to '{table_name}'.")
            return df

        except Exception as e:
            logging.error(f"❌ Failed to update woodMaster paths: {e}", exc_info=True)
            return None


