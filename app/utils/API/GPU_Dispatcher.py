from fastapi import APIRouter, Request
from datetime import datetime, timezone
import logging
import os
import random

GPUDispatcherRouter = APIRouter()

# In-memory worker registry
from app.tasks.TA03_GPUDispatcher.utils.worker_registry import worker_registry

WORKER_BLACKLIST = []




@GPUDispatcherRouter.post("/worker/register")
async def register_worker(request: Request):
    data = await request.json()
    worker_name = data.get("worker_name")
    # No need for `ip` anymore

    if not worker_name:
        return {"status": "error", "detail": "worker_name required"}

    if worker_name in WORKER_BLACKLIST:
        return {"status": "error", "detail": f"Worker {worker_name} is blacklisted"}

    worker_registry[worker_name] = {
        "last_seen": datetime.now(timezone.utc),
        "start_time": data.get("start_time"),
        "queue_length": data.get("queue_length", 0),
        "queued_ids": set(data.get("queued_ids", [])),
        "running_ids": set(data.get("running_ids", [])),
        "finished_ids": set(data.get("finished_ids", [])),  # cumulative
        "successfull_ids": set(data.get("successful_ids", [])),  # cumulative
        "allow_jobs": data.get("allow_jobs", True),
        "last_time_job_added": data.get("last_time_job_added"),
        "current_version": data.get("current_version", "unknown"),
    }


    if worker_name not in worker_registry:
        logging.info(f"🔔 Worker {worker_name} registered")
    else:
        logging.debug1(f"🔔 Worker {worker_name} re-registered")
    if worker_registry[worker_name].get("current_version") != os.getenv("WORKER_VERSION", "unknown"):
        logging.warning(f"⚠️ Worker {worker_name} version mismatch: {worker_registry[worker_name].get('current_version')} != {os.getenv('WORKER_VERSION', 'unknown')}")
        worker_registry[worker_name]["allow_jobs"] = False
        WORKER_BLACKLIST.append(worker_name)

    if random.random() < 0.95:
        WORKER_BLACKLIST.clear()  # Clear blacklist with 95% probability
        logging.debug2("🔄 Worker blacklist cleared")

    

    return {"status": "registered", "worker_name": worker_name}
