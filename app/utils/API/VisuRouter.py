from fastapi import APIRouter, Query
from sqlalchemy import inspect
from app.utils.SQL.DBEngine import DBEngine
import pandas as pd
import numpy as np

from pydantic import BaseModel

router = APIRouter()

# --- Pydantic model for POST /visu/data ---
class DataFetchRequest(BaseModel):
    db_key: str
    table: str
    limit: int = 1000

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
        sql = f'SELECT * FROM "{req.table}" LIMIT {req.limit}'
        df = pd.read_sql(sql, engine)

        # ✅ Replace NaN/Infinity with None so it’s JSON-compliant
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})

        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

