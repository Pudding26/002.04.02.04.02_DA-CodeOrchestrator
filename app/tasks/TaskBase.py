from abc import ABC, abstractmethod
import uuid
import io
from datetime import datetime
from app.utils.controlling.TaskController import TaskController
from app.utils.SQL.DBEngine import DBEngine
from app.utils.logger.ProfileLogger import ProfileLogger 
from sqlalchemy.orm import Session

class TaskBase(ABC):
    """
    Abstract base class for all tasks. Enforces a standard lifecycle and integrates task control.
    """

    def __init__(self, instructions: dict, controller: TaskController, enable_profiling: bool = True):
        """
        :param instructions: Dictionary of task-specific parameters
        :param controller: An instance of TaskController (DB-backed for progress/control)
        """
        self.instructions = instructions
        self.controller = controller or TaskController(
            task_name=self.instructions.get("taskName"),
            task_uuid=str(self.task_uuid)
            )
        
        self.status = "Initialized"
        self.task_uuid = self.controller.task_uuid
        self.enable_profiling = enable_profiling
        self._profiler_streams = {}
        

        if self.enable_profiling:
            self._setup_memory_profiling()
        
        
        self.setup()

    def _setup_memory_profiling(self):
        """Prepare memory streams and ORM handler."""
        self.device = "Lele_Lenovo"
        self.profile_type = "memProfile"
        self.task_group = self.instructions.get("task_group", "UnknownGroup")
        self.task_name = self.instructions.get("taskName") or self.__class__.__name__

        # Create in-memory streams per step
        self._profiler_streams = {
            "step1": io.StringIO(),
            "step2": io.StringIO(),
            "step3": io.StringIO()
        }

        # Create DB session
        session: Session = DBEngine("progress").get_session()

        # ORM profiler handler
        self.ProfileLogger = ProfileLogger(
            task_group=self.task_group,
            task_name=self.task_name,
            device=self.device,
            session=session,
            profile_type=self.profile_type
        )
        self.ProfileLogger.task_uuid = self.task_uuid  # assign shared UUID

    def flush_memory_logs(self):
        """Push all memory logs to database."""
        if not self.enable_profiling:
            return
        for stream in self._profiler_streams.values():
            self.ProfileLogger.log_stream_to_db(stream)






    @abstractmethod
    def setup(self):
        """
        Setup method to prepare resources, configurations, or connections.
        Must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def run(self):
        """
        The main logic of the task. Must be implemented by the subclass.
        This method should call self.check_control() periodically.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up resources. Called at the end of task execution or after error.
        """
        pass

    def check_control(self):
        """
        Check if the task should pause or stop.
        Should be called periodically in long-running loops.
        """
        self.controller.wait_if_paused()
        if self.controller.should_stop():
            self.status = "Stopped"
            raise InterruptedError("Task was stopped.")
