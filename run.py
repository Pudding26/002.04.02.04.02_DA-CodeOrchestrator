import uvicorn
from dotenv import load_dotenv
import os
import logging

from app.utils.logger.UvicornLoggingFilter import LOGGING_CONFIG  




if __name__ == "__main__":
    load_dotenv()
    BACKEND_ORCH_PRIVATE_PORT = int(os.getenv("BACKEND_ORCH_PRIVATE_PORT", 8000))
    
    logging.info("CWD =", os.getcwd())
    print("HELLO")



    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=BACKEND_ORCH_PRIVATE_PORT,
        reload=True,
        log_config=LOGGING_CONFIG
    )
