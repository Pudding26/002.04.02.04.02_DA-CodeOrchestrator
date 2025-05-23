from pydantic import BaseModel

class Task1Request(BaseModel):
    task_name: str
    parameter_a: int
    parameter_b: str

class Task1Response(BaseModel):
    status: str
    task: str
    result: str  # any additional result details for Task 1
