def generate_doe_uuid(row: dict) -> str:
    """
    Generates a deterministic DoE_UUID from a config row.
    - Sorts column keys alphabetically.
    - Sorts values inside each column.
    - Stringifies nested lists.
    """
    import hashlib

    normalized_row = {}

    for k in sorted(row.keys()):
        v = row[k]

        if isinstance(v, list):
            # Normalize elements: stringify sublists
            norm_v = [str(item) if isinstance(item, list) else item for item in v]
            # Sort normalized values
            sorted_v = sorted(norm_v)
            normalized_row[k] = sorted_v
        else:
            normalized_row[k] = [v]

    # Sort the entire normalized row by keys (already iterated sorted)
    normalized_items = list(normalized_row.items())  # Already sorted due to for-loop order

    base_str = str(normalized_items)
    return "DoE_" + hashlib.sha1(base_str.encode()).hexdigest()[:10]
