from fastapi import APIRouter
from app.tasks.TaskWrapper import TaskRequestBase as TaskRequest
from app.utils.API.TaskHandler import TaskHandler

router = APIRouter()
handler = TaskHandler()

@router.get("/")
def get_tasks():
    return {"possible_tasks": handler.get_tasks()}

@router.post("/start")
def start_task(req: TaskRequest):
    return handler.start_task(req.task_name)

@router.post("/stop")
def stop_task(req: TaskRequest):
    return handler.stop_task(req.task_name)
