from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from threading import Timer
import time

from app.utils.logger.loggingWrapper import LoggingHandler
from app.utils.API.TaskRouter import router as task_router
from app.utils.API.VisuRouter import router as visu_router

from fastapi.middleware.cors import CORSMiddleware

from app.tasks.TA01_setup.TA01_A_SQLSetup import TA01_A_SQLSetup
from app.tasks.TA01_setup.TA01_B_PIDCleaup import TA01_B_PIDCleaup

from app.utils.HDF5.HDF5Utils import HDF5Utils
from app.utils.controlling.TaskController import TaskController
from app.tasks.TaskBase import TaskBase

from app.tasks.TA02_HDF5Storer.TA02_0_HDF5Storer import TA02_0_HDF5Storer


logger = logging.getLogger(__name__)
LoggingHandler(logging_level="DEBUG-2")


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    TA01_A_SQLSetup.createDatabases()
    TA01_A_SQLSetup.create_all_tables()
    TA01_B_PIDCleaup.cleanup_pids()

    TaskController.clean_orphaned_tasks_on_start()


    

    directories = [
    "data/rawData/",
    "data/productionData/",
    ]
    
    # ✅ This block runs on startup
    logger.info("🔓 Unlocking dirty HDF5 files...")
    HDF5Utils.unlock_dirty_hdf5_files(directories=directories)

    # Defer the HTTP-based task trigger until the app is actually serving
    def delayed_trigger():
        time.sleep(5)  # Give Uvicorn a moment to be fully serving
        logger.info("🚀 Triggering TA02_0_HDF5Storer via HTTP")
        try:
            TaskBase.trigger_task_via_http("TA02_0_HDF5Storer")
        except Exception as e:
            logger.error(f"❌ Failed to start TA02_0_HDF5Storer: {e}")

    Timer(0, delayed_trigger).start()

    yield



app = FastAPI(lifespan=lifespan)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(task_router, prefix="/tasks", tags=["tasks"])
app.include_router(visu_router, prefix="/visu")

@app.get("/")
def read_root():
    return {"message": "Orchestrator is running"}

