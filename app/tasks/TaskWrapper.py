from pydantic import BaseModel

class TaskRequestBase(BaseModel):
    task_name: str

class TaskResponseBase(BaseModel):
    status: str
    task: str
