import pandas as pd
import logging

def split_df_based_on_max_split(df: pd.DataFrame, column: str, separator: str = '_') -> dict:
    """
    Splits a DataFrame into multiple sub-DataFrames based on the number of parts after splitting a target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Column to split by.
    - separator (str): Delimiter used to split the column string.

    Returns:
    - dict[int, pd.DataFrame]: Dictionary where keys are split lengths, and values are DataFrames.
    """
    split_dict = {}
    try:
        for idx, row in df.iterrows():
            val = row[column]
            if not isinstance(val, str):
                continue
            parts = val.split(separator)
            max_split = len(parts)
            row_data = row.copy()
            row_data["max_split"] = max_split
            for i, part in enumerate(parts):
                row_data[f"col-{i}"] = part

            if max_split not in split_dict:
                split_dict[max_split] = []
            split_dict[max_split].append(row_data)

        # Convert lists to DataFrames
        for key in split_dict:
            split_dict[key] = pd.DataFrame(split_dict[key])

        logging.debug1(f"🔍 Split column '{column}' into {len(split_dict)} groups by max_split.")
        return split_dict

    except Exception as e:
        logging.error(f"❌ Error splitting DataFrame on column '{column}': {e}")
        raise
