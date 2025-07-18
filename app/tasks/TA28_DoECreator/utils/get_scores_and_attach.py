import yaml
import pandas as pd
from sqlalchemy import select
from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.models.production.orm_ModellingResults import orm_ModellingResults
from app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out


def get_scores_and_attach(doe_df, acc_type="rf_acc", theoretical_max_size=10000, gen = -1, subbranch_id=None):
    """
    Attach weighted average scores (accuracy, sample size, entropy) to each DoE_UUID in doe_df.

    For each DoE_UUID, the following weighted averages are computed:

    - score_acc: Weighted average of the selected accuracy metric (acc_type) 
      using LABEL_WEIGHTS and FRAQ_WEIGHTS.
    
    - score_sample: Weighted average of normalized initial_row_count 
      (divided by theoretical_max_size), weighted by LABEL_WEIGHTS and FRAQ_WEIGHTS.
    
    - score_entropy: Weighted average of 1 - normalized entropy across configured fields 
      (using ENTROPY_WEIGHTS), then weighted by LABEL_WEIGHTS and FRAQ_WEIGHTS.

    Args:
        doe_df (DataFrame): Input DataFrame containing a 'DoE_UUID' column.
        acc_type (str): Accuracy field to use for score_acc (e.g., 'rf_acc', 'knn_acc').
        theoretical_max_size (int): Max sample size used for normalizing initial_row_count.

    Returns:
        DataFrame: The input DataFrame with three additional columns:
            - 'score_acc'
            - 'score_sample'
            - 'score_entropy'

    Notes:
        - All weighting schemes are loaded from weights.yaml.
        - If all metric values equal 1.0, scores will normalize to 1.0 
          (due to weighted average structure).
    """

    # Load weights.yaml
    with open("app/config/DoE/weights.yaml", "r") as f:
        weights = yaml.safe_load(f)

    label_weights = weights.get("LABEL_WEIGHTS", {})
    fraq_weights = weights.get("FRAQ_WEIGHTS", {})
    entropy_weights = weights.get("ENTROPY_WEIGHTS", {})

    entropy_fields = [f"{label}_entropy_train" for label in entropy_weights.keys()]
    doe_df.rename(columns={"job_uuid": "DoE_UUID"}, inplace=True)
    doe_uuids = doe_df["DoE_UUID"].unique().tolist()

    session = DBEngine(db_key="production").get_session()
    try:
        stmt = select(orm_ModellingResults).where(
            orm_ModellingResults.DoE_UUID.in_(doe_uuids)
        )
        results = session.execute(stmt).scalars().all()
        records = [ModellingResults_Out.model_validate(res).model_dump() for res in results]
        df_results = pd.DataFrame(records)


        # Initialize running sums for weighted averages
        acc_weighted_sum = {}
        acc_weight_sum = {}

        size_weighted_sum = {}
        size_weight_sum = {}

        entropy_weighted_sum = {}
        entropy_weight_sum = {}

        for res in results:
            uuid = res.DoE_UUID
            fraq = float(getattr(res, "frac", None))
            label = getattr(res, "label", None)

            fraq_weight = fraq_weights.get(fraq, 0)
            label_weight = label_weights.get(label, 0)
            w = fraq_weight * label_weight

            if w == 0:
                continue  # Skip this row if no weight contribution

            # 1️⃣ ACCURACY weighted contribution
            acc_val = getattr(res, acc_type, 0) or 0
            acc_weighted_sum[uuid] = acc_weighted_sum.get(uuid, 0) + w * acc_val
            acc_weight_sum[uuid] = acc_weight_sum.get(uuid, 0) + w

            # 2️⃣ SAMPLE SIZE weighted contribution
            init_row_count = getattr(res, "initial_row_count", 0) or 0
            normalized_size = init_row_count / theoretical_max_size
            size_weighted_sum[uuid] = size_weighted_sum.get(uuid, 0) + w * normalized_size
            size_weight_sum[uuid] = size_weight_sum.get(uuid, 0) + w

            # 3️⃣ ENTROPY weighted contribution
            weighted_entropy_sum = 0
            total_entropy_weight = 0
            for field, e_weight in entropy_weights.items():
                field_name = f"{field}_entropy_train"
                entropy_val = getattr(res, field_name, None)
                if entropy_val is not None:
                    weighted_entropy_sum += entropy_val * e_weight
                    total_entropy_weight += e_weight

            if total_entropy_weight > 0:
                normalized_entropy = weighted_entropy_sum / total_entropy_weight
            else:
                normalized_entropy = 0

            entropy_contrib = normalized_entropy
            entropy_weighted_sum[uuid] = entropy_weighted_sum.get(uuid, 0) + w * entropy_contrib
            entropy_weight_sum[uuid] = entropy_weight_sum.get(uuid, 0) + w

        # Safe division utility
        def safe_div(numerator, denominator):
            return numerator / denominator if denominator != 0 else 0

        doe_df["score_acc"] = doe_df["DoE_UUID"].apply(
            lambda uuid: safe_div(acc_weighted_sum.get(uuid, 0), acc_weight_sum.get(uuid, 0))
        )
        doe_df["score_sample"] = doe_df["DoE_UUID"].apply(
            lambda uuid: safe_div(size_weighted_sum.get(uuid, 0), size_weight_sum.get(uuid, 0))
        )
        doe_df["score_entropy"] = doe_df["DoE_UUID"].apply(
            lambda uuid: safe_div(entropy_weighted_sum.get(uuid, 0), entropy_weight_sum.get(uuid, 0))
        )

        summary_records = []
        for _, row in doe_df.iterrows():
            uuid = row["DoE_UUID"]
            summary_records.append({
                "DoE_UUID": uuid,
                "acc_type": acc_type,
                "sub_branch_id": subbranch_id,
                "gen": gen,
                "score_acc": safe_div(acc_weighted_sum.get(uuid, 0), acc_weight_sum.get(uuid, 0)),
                "score_sample": safe_div(size_weighted_sum.get(uuid, 0), size_weight_sum.get(uuid, 0)),
                "score_entropy": safe_div(entropy_weighted_sum.get(uuid, 0), entropy_weight_sum.get(uuid, 0)),
                "acc_weight_sum": acc_weight_sum.get(uuid, 0),
                "sample_weight_sum": size_weight_sum.get(uuid, 0),
                "entropy_weight_sum": entropy_weight_sum.get(uuid, 0)
            })

        df_summary = pd.DataFrame(summary_records)






        return doe_df, df_summary

    except Exception as e:
        print(f"❌ get_scores_and_attach() failed: {e}")
        # Set all scores to 0 fallback:
        doe_df["score_acc"] = 0
        doe_df["score_sample"] = 0
        doe_df["score_entropy"] = 0
        return doe_df

    finally:
        session.close()
