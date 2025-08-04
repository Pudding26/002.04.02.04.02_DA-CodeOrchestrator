# app/utils/SQL/RetryingSession.py
import logging, time, random
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import psycopg2

class RetryingSession(Session):
    """Custom Session that retries on deadlocks with exponential backoff."""

    def commit(self, retries=5, base_delay=1, max_delay=30):
        for attempt in range(retries):
            try:
                return super().commit()
            except OperationalError as e:
                # Check if this is a PostgreSQL deadlock
                if isinstance(e.orig, psycopg2.errors.DeadlockDetected):
                    wait = min(base_delay * 2**attempt + random.random(), max_delay)
                    logging.warning(
                        f"⚠️ Deadlock detected on commit, retry {attempt+1}/{retries} "
                        f"in {wait:.2f}s..."
                    )
                    self.rollback()  # rollback before retry
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("❌ Transaction failed after retries due to deadlock")
