



class DataDebugValidator:

    def find_columns_unique_to_one(data, groupby_cols=None):
        """
        Find columns that appear in only one group or one dataframe.

        Parameters:
        - data: dict of DataFrames OR a single DataFrame
        - groupby_cols: if data is a DataFrame, this list is used to group it into sub-DataFrames

        Returns:
        - dict with:
        - 'unique_columns': {column: [group_keys]}
        - 'only_in_one': list of columns only found in one group
        """
        from collections import defaultdict

        # If a single DataFrame is provided with groupby cols, split it into a dict
        if isinstance(data, pd.DataFrame):
            if not groupby_cols:
                raise ValueError("If providing a DataFrame, you must also provide groupby_cols")
            grouped = data.groupby(groupby_cols)
            df_dict = {str(name): group for name, group in grouped}
        elif isinstance(data, dict):
            df_dict = data
        else:
            raise TypeError("data must be a DataFrame or a dict of DataFrames")

        col_sources = defaultdict(list)

        for name, df in df_dict.items():
            for col in df.columns:
                col_sources[col].append(name)

        unique_cols = {col: sources for col, sources in col_sources.items() if len(sources) == 1}
        only_in_one = list(unique_cols.keys())

        return {
            "unique_columns": unique_cols,
            "only_in_one": only_in_one,
        }

    


    def count_fake_nans(data, fake_nans={"NaN", "nan", ""}, groupby_col=None):
        """
        Count fake NaNs per column. Works on a single DataFrame or a dict of DataFrames.

        Parameters:
        - data: pd.DataFrame or dict of DataFrames
        - fake_nans: set of values to treat as fake NaNs (default: {"NaN", "nan", ""})
        - groupby_col: optional column name to group by (e.g., "sourceNo")

        Returns:
        - pd.DataFrame with counts per column (and per group if specified)
        """
        import pandas as pd
        from collections import defaultdict

        def _count_in_df(df):
            if groupby_col and groupby_col in df.columns:
                grouped = df.groupby(groupby_col)
                result = defaultdict(dict)
                for group_val, group_df in grouped:
                    for col in group_df.columns:
                        result[group_val][col] = group_df[col].isin(fake_nans).sum()
                return pd.DataFrame(result).T  # transpose to match group/column format
            else:
                return pd.Series({col: df[col].isin(fake_nans).sum() for col in df.columns})

        if isinstance(data, dict):
            result_dict = {}
            for name, df in data.items():
                result_dict[name] = _count_in_df(df)
            return result_dict  # return dict of Series or DataFrames
        elif isinstance(data, pd.DataFrame):
            return _count_in_df(data)
        else:
            raise TypeError("Input must be a DataFrame or a dict of DataFrames.")

