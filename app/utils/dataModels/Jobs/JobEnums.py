from enum import Enum   

class JobStatus(str, Enum):
    TODO = "todo"
    DONE = "done"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"



class RelationState(str, Enum):
    FREE        = "free"         # child finished successfully
    IN_PROGRESS = "in_progress"  # child still running
    BLOCKED     = "blocked"      # waiting for something else
    FAILED      = "failed"       # child failed


class JobKind(str, Enum):
    PROVIDER   = "provider"
    SEGMENTER  = "segmenter"
    MODELER    = "modeler"
    TRANSFER   = "transfer"
    DOE       = "DoE"