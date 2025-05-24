import os
import yaml
import logging
import pandas as pd


class YamlColumnMapper:
    """Handles column renaming and static column injection from YAML files."""

    @staticmethod
    def rename_columns(df: pd.DataFrame, yaml_path: str, keys_list=None) -> pd.DataFrame:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        section = yaml_data
        if keys_list:
            for key in keys_list:
                section = section.get(key, {})

        if not isinstance(section, dict):
            raise ValueError(f"YAML section at {' -> '.join(keys_list)} is not a dictionary.")

        return df.rename(columns=section)

    @staticmethod
    def add_static_columns(df: pd.DataFrame, yaml_path: str, keys_list: list) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            section = yaml_data
            for key in keys_list:
                section = section.get(key, {})

            if not isinstance(section, dict):
                raise ValueError(f"Expected dictionary at {' -> '.join(keys_list)}, got: {type(section)}")

            for col, val in section.items():
                df[col] = val

            logger.info(f"✅ Injected {len(section)} static columns.")
            return df

        except Exception as e:
            logger.warning(f"YAML column injection failed: {e}")
            return df

    @staticmethod
    def update_column_mapping(df: pd.DataFrame, yaml_path: str, keys_list: list, default_value="TODO"):
        logger = logging.getLogger(__name__)
        yaml_data = {}

        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        current_section = yaml_data
        for key in keys_list:
            current_section.setdefault(key, {})
            current_section = current_section[key]

        added = 0
        for col in df.columns:
            if col not in current_section:
                current_section[col] = default_value
                added += 1

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

        logger.debug(f"📝 Updated YAML with {added} new column mappings under {' -> '.join(keys_list)}.")



    @staticmethod
    def yaml_col_value_mapper(yaml_path: str, data_source_key: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies value mappings from a YAML configuration to the given DataFrame.

        Args:
            yaml_path (str): Path to the unified YAML file.
            data_source_key (str): Key to identify the data source in the YAML file (e.g., 'DS01').
            df (pd.DataFrame): DataFrame to process.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger = logging.getLogger(__name__)
        logger.debug2(f"🔍 Loading value mappings from: {yaml_path}")
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

        with yaml_file.open("r") as file:
            try:
                mappings = yaml.safe_load(file)
                logger.debug2(f"✅ YAML parsed successfully. Top-level keys: {list(mappings.keys())}")
            except yaml.YAMLError as e:
                logger.error(f"❌ Failed to parse YAML: {e}")
                raise

        if data_source_key not in mappings:
            logger.warning(f"⚠️ No mapping found for key: {data_source_key}")
            return df

        source_mappings = mappings[data_source_key]

        # Normalize to list of mappings if necessary
        if isinstance(source_mappings, dict) and "col" in source_mappings:
            source_mappings = [source_mappings]
        elif isinstance(source_mappings, dict):
            # Handle dict-of-dicts
            source_mappings = [v for k, v in source_mappings.items() if isinstance(v, dict) and "col" in v]

        for mapping in source_mappings:
            col = mapping.get("col")
            col_old = mapping.get("col_old")
            values = mapping.get("values", {})

            if col not in df.columns:
                logger.debug2(f"⏭️ Column '{col}' not found in DataFrame. Skipping.")
                continue

            logger.debug2(f"🔁 Processing mapping for column: {col}")
            if col_old:
                df[col_old] = df[col].copy()
                logger.debug2(f"💾 Backed up original column to: {col_old}")

            df[col] = df[col].map(values).fillna(df[col])  # fallback to original if no mapping

            # Auto rename todo_ columns
            if col.startswith("todo_"):
                new_col = col.replace("todo_", "")
                df.rename(columns={col: new_col}, inplace=True)
                logger.debug2(f"🔤 Renamed column '{col}' → '{new_col}'")

        logger.debug2(f"✅ Value mapping applied successfully for key: {data_source_key}")
        return df