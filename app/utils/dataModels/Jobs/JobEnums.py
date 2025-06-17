from enum import Enum   

class JobStatus(str, Enum):
    TODO = "todo"
    DONE = "done"
    FAILED = "failed"