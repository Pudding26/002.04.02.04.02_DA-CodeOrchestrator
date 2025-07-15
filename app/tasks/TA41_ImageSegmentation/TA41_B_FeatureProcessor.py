from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import pandas as pd


class TA41_B_FeatureProcessor:
    _BIN_FEATURES = ["area", "eccentricity", "major_axis_length"]
    _SUM_FEATS = [
        "area",
        "eccentricity",
        "major_axis_length",
        #"solidity",
        #"orientation",
        "perimeter",
        "equivalent_diameter",
        "minor_axis_length",
        "extent",
        #"euler_number",
    ]
   #     + [f"moments_hu-{i}" for i in range(7)]
    _STATS = ["mean", "std", "min", "max"]
    _UNITS = {
        "area": "px²",
        "eccentricity": "-",
        "major_axis_length": "px",
        "solidity": "-",
        "orientation": "°",
    }

    def process(self, shotID: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        summary_df = self._summarise(df)
        summary_df["shotID"] = shotID
        return summary_df

    def process_all(self, stackID, dfs) -> pd.DataFrame:
        all_rows = []
        for i, df in enumerate(dfs):
            shot_id = f"{stackID}_{i+1:03d}"
            summary_df = self.process(shot_id, df)
            summary_df["stackID"] = stackID
            if not summary_df.empty:
                all_rows.append(summary_df)

        feature_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        
        return feature_df


    def _summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        all_rows = []
        count_rows = []
        n_bins = 4
        edges = np.linspace(0, 1, n_bins + 1, dtype=np.float32)
        percentiles = (edges * 100).astype(int)
        bin_labels = [f"p{percentiles[i]:02d}-p{percentiles[i+1]:02d}" for i in range(n_bins)]

        for anchor in self._BIN_FEATURES:
            if anchor not in df.columns:
                continue
            quantile_edges = np.unique(df[anchor].quantile(edges, interpolation="nearest").to_numpy())
            if quantile_edges.size < 2:
                continue

            bins = pd.cut(df[anchor], bins=quantile_edges,
                        labels=bin_labels[: len(quantile_edges)-1],
                        include_lowest=True, right=True)

            bin_type = f"by_{anchor}"

            # Calculate and store bin count info once per anchor
            bin_counts = df.groupby(bins, observed=False).size().reset_index(name="bin_count")
            bin_counts.rename(columns={bin_counts.columns[0]: "bin_label"}, inplace=True)
            bin_counts["bin_fraction"] = bin_counts["bin_count"] / len(df)
            bin_counts["bin_type"] = bin_type
            count_rows.append(bin_counts)

            # Feature summaries
            target_cols = [c for c in self._SUM_FEATS if c in df.columns]
            agg_funcs = {c: self._STATS for c in target_cols}
            tbl = df.groupby(bins, observed=False)[target_cols].agg(agg_funcs)
            tbl.columns = [f"{c}_{stat}" for c, stat in tbl.columns]
            tbl.reset_index(inplace=True)
            tbl.rename(columns={tbl.columns[0]: "bin_label"}, inplace=True)

            skew_df = df.groupby(bins, observed=False)[target_cols].apply(lambda g: g.skew()).add_suffix("_skew")
            skew_df.reset_index(inplace=True)
            skew_df.rename(columns={skew_df.columns[0]: "bin_label"}, inplace=True)

            tbl = pd.merge(tbl, skew_df, on="bin_label", how="left")

            long_df = tbl.melt(
                id_vars=["bin_label"],
                var_name="feature_stat",
                value_name="feature_value"
            )
            long_df[["feature_name", "stat_type"]] = long_df["feature_stat"].str.rsplit("_", n=1, expand=True)
            long_df.drop(columns="feature_stat", inplace=True)
            long_df["bin_type"] = bin_type
            long_df["unit"] = long_df["feature_name"].map(self._UNITS).astype("category")

            all_rows.append(long_df)

        # Combine all rows
        df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

        # Merge bin count info once
        if count_rows and not df.empty:
            count_df = pd.concat(count_rows, ignore_index=True)
            df = pd.merge(df, count_df, on=["bin_label", "bin_type"], how="left")

        return df

