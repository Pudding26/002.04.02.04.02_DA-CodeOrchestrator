import logging

def apply_must_rules(df, params, field):
    """
    Apply must_contain / must_not_contain logic to filter a DataFrame column.

    Semantics:
    - must_contain: values that must be present
    - must_contain == ["ALL"]: require *all values* of this field
    - must_not_contain: forbidden values
    - must_not_contain == ["ALL"]: forbid everything except must_contain
    """
    must_contain = params.get("must_contain", []) or []
    must_not_contain = params.get("must_not_contain", []) or []

    # --- Expand must_contain == ["ALL"] ---------------------
    if must_contain == ["ALL"]:
        if field in df.columns:
            must_contain = df[field].dropna().unique().tolist()
            logging.debug2("Case: field=%s must_contain=ALL → expanded to all values", field)
            logging.debug1("Expanded must_contain list for %s: %s", field, must_contain)
        else:
            must_contain = []

    # --- Expand must_not_contain == ["ALL"] -----------------
    if must_not_contain == ["ALL"]:
        if field in df.columns:
            valid_options = df[field].dropna().unique().tolist()
            must_not_contain = [v for v in valid_options if v not in must_contain]
            logging.debug2("Case: field=%s must_not_contain=ALL → forbidding everything except must_contain", field)
            logging.debug1("Expanded must_not_contain list for %s: %s", field, must_not_contain)

    # --- Apply must_contain -----------------
    if must_contain and field in df.columns:
        before = len(df)
        df = df[df[field].isin(must_contain)]
        logging.debug2("Applied must_contain on %s; rows before=%d after=%d", field, before, len(df))
        logging.debug1("must_contain values applied for %s: %s", field, must_contain)

    # --- Apply must_not_contain -----------------
    if must_not_contain and field in df.columns:
        before = len(df)
        df = df[~df[field].isin(must_not_contain)]
        logging.debug2("Applied must_not_contain on %s; rows before=%d after=%d", field, before, len(df))
        logging.debug1("must_not_contain values applied for %s: %s", field, must_not_contain)

    return df
