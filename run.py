import uvicorn
from dotenv import load_dotenv
import os
import logging

from app.utils.logger.UvicornLoggingFilter import LOGGING_CONFIG  




if __name__ == "__main__":
    load_dotenv()
    BACKEND_ORCH_BASE_URL = os.getenv("BACKEND_ORCH_BASE_URL")
    BACKEND_ORCH_BASE_PORT = int(os.getenv("BACKEND_ORCH_BASE_PORT"))
    logging.info("CWD =", os.getcwd())

    uvicorn.run(
        "app.main:app",
        host=BACKEND_ORCH_BASE_URL,
        port=BACKEND_ORCH_BASE_PORT,
        reload=True,
        log_config=LOGGING_CONFIG
    )
