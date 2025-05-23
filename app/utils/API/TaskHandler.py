from fastapi import HTTPException

class TaskHandler:
    def __init__(self):
        # This could later be loaded from a DB, instruction file, etc.
        self.task_list = ["DT001", "DT002", "DT003"]

    def get_tasks(self):
        return self.task_list

    def start_task(self, task_name: str):
        if task_name not in self.task_list:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # TODO: Connect to your real ImportHandler logic
        print(f"🚀 Starting task: {task_name}")
        return {"status": "started", "task": task_name}

    def stop_task(self, task_name: str):
        if task_name not in self.task_list:
            raise HTTPException(status_code=404, detail="Task not found")

        print(f"🛑 Stopping task: {task_name}")
        return {"status": "stopped", "task": task_name}
