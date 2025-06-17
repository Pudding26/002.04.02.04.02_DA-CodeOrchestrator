from pydantic import BaseModel, Field
import datetime as dt

class RetryInfo(BaseModel):
    attempts    : int          = 0            # how many times the job was popped
    last_error  : str | None   = None         # latest failure note
    penalty     : int          = 0            # the “cost” your scheduler uses
    next_retry  : dt.datetime  = Field(
        default_factory=lambda: dt.datetime.utcnow()
    )

    def register_failure(self, error: str,
                         penalty_step: int = 1,
                         backoff: float = 1.5) -> None:
        """
        • Increase attempts + penalty
        • Push next_retry into the future  (simple exponential back-off)
        """
        self.attempts   += 1
        self.penalty    += penalty_step
        self.last_error  = error

        # exponential : now + penalty * backoff ^ attempts   (tweak if needed)
        delay_seconds = int(self.penalty * (backoff ** self.attempts))
        self.next_retry = dt.datetime.utcnow() + dt.timedelta(seconds=delay_seconds)
