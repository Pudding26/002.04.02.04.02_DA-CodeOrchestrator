from fastapi import APIRouter, Query
from sqlalchemy import inspect
from app.utils.SQL.DBEngine import DBEngine
import pandas as pd
import numpy as np
from sqlalchemy import text



from app.tasks.TA01_setup.TA01_C_SQLMetaCache import TA01_C_SQLMetaCache

from pydantic import BaseModel

router = APIRouter()

# --- Pydantic model for POST /visu/data ---
class DataFetchRequest(BaseModel):
    db_key: str
    table: str
    limit: int = 1000
    limit_mode: str = "top"  # 'top', 'bottom', 'random'
    sort_col: str | None = None
    sort_order: str = "desc"  # 'asc' or 'desc'
    filters: dict = {}  # {col_name: filter_spec}


# --- GET /visu/tables ---
@router.get("/tables")
def list_tables(db_key: str = Query(..., description="Environment key for DB (e.g., 'raw')")):
    engine = DBEngine(db_key).get_engine()
    inspector = inspect(engine)
    return {"tables": inspector.get_table_names()}

# --- GET /visu/columns ---
@router.get("/columns")
def get_columns(db_key: str, table: str):
    engine = DBEngine(db_key).get_engine()
    inspector = inspect(engine)
    columns = inspector.get_columns(table)
    return {
        "columns": [
            {"name": col["name"], "type": str(col["type"])}
            for col in columns
        ]
    }

# --- POST /visu/data ---

@router.post("/data")
def fetch_data(req: DataFetchRequest):
    try:
        engine = DBEngine(req.db_key).get_engine()
        
        base_query = f'SELECT * FROM "{req.table}"'
        where_clauses = []

        # Build SQL filters:
        for col, spec in req.filters.items():
            if spec["type"] == "numeric":
                min_val = spec.get("min")
                max_val = spec.get("max")
                if min_val is not None:
                    where_clauses.append(f'"{col}" >= {min_val}')
                if max_val is not None:
                    where_clauses.append(f'"{col}" <= {max_val}')
            elif spec["type"] == "categorical":
                values = spec.get("values", [])
                if len(values) > 0:
                    clauses = [f'"{col}" ILIKE \'%{val}%\'' for val in values]
                    where_clauses.append(f"({' OR '.join(clauses)})")
        
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        order_sql = ""
        if req.limit_mode in ["top", "bottom"] and req.sort_col:
            order = "ASC" if req.limit_mode == "bottom" else "DESC"
            order_sql = f' ORDER BY "{req.sort_col}" {order}'

        # NOTE: Random limiting handled here
        if req.limit_mode == "random":
            order_sql = " ORDER BY RANDOM()"

        limit_sql = f" LIMIT {req.limit}"

        final_sql = base_query + where_sql + order_sql + limit_sql

        df = pd.read_sql(final_sql, engine)
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/filterschema")
def get_filter_schema(db_key: str, table: str):
    engine = DBEngine(db_key).get_engine()
    insp = inspect(engine)
    cols_info = insp.get_columns(table)
    metadata = []

    with engine.connect() as conn:
        for col in cols_info:
            col_name = col["name"]
            col_type = str(col["type"])

            if "INTEGER" in col_type or "FLOAT" in col_type or "NUMERIC" in col_type:
                result = conn.execute(text(f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM "{table}"'))
                min_val, max_val = result.fetchone()
                metadata.append({
                    "name": col_name,
                    "type": "numeric",
                    "min": min_val,
                    "max": max_val
                })
            else:
                result = conn.execute(text(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}"'))
                unique_count = result.scalar()
                if unique_count <= 10:
                    result = conn.execute(text(f'SELECT DISTINCT "{col_name}" FROM "{table}" LIMIT 10'))
                    values = [row[0] for row in result.fetchall()]
                    metadata.append({
                        "name": col_name,
                        "type": "categorical",
                        "unique_values": values
                    })
                else:
                    metadata.append({
                        "name": col_name,
                        "type": "categorical",
                        "unique_count": unique_count
                    })

    return {"columns": metadata}


class TableCacheRequest(BaseModel):
    db_key: str
    table_name: str

@router.post("/cache/update_table_cache")
def update_table_cache(req: TableCacheRequest):
    success = TA01_C_SQLMetaCache.update_cache(req.db_key, req.table_name)
    if success:
        return {"status": "updated", "db_key": req.db_key, "table_name": req.table_name}
    else:
        raise HTTPException(status_code=500, detail="Failed to update table cache")

@router.get("/cache/all_table_caches")
def get_all_table_caches():
    result = TA01_C_SQLMetaCache.get_all_table_caches()
    return result