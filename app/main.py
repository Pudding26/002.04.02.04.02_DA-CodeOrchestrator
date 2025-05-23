from fastapi import FastAPI
from pydantic import BaseModel

from app.utils.API.TaskRouter import router as task_router

app = FastAPI()
app.include_router(task_router, prefix="/tasks", tags=["tasks"])

class JobRequest(BaseModel):
    input: str

@app.get("/")
def read_root():
    return {"message": "Orchestrator is running"}
