import os, signal, atexit
import logging

class TA01_B_PIDCleaup:
    PIDFILE = '/tmp/orchestrator_PIDs.txt'
    _atexit_registered = False

    @classmethod
    def cleanup_pids(cls) -> None:
        """Legacy method for cleaning up stale PIDs on startup."""
        if not os.path.exists(cls.PIDFILE):
            logging.debug2("No PID file found — nothing to clean up.")
            return

        cls._cleanup_pidfile()

    @classmethod
    def _cleanup_pidfile(cls) -> None:
        attempted = killed = failed = 0

        with open(cls.PIDFILE) as f:
            pids = [line.strip() for line in f if line.strip()]

        attempted = len(pids)
        for pid_str in pids:
            try:
                pid = int(pid_str)
                os.kill(pid, signal.SIGTERM)
                killed += 1
                logging.debug2(f"Killed process with PID {pid}.")
            except ProcessLookupError:
                failed += 1
                logging.debug2(f"No process with PID {pid} found — might have already stopped.")
            except Exception as e:
                failed += 1
                logging.exception(f"Error killing process {pid}: {e}")

        try:
            os.remove(cls.PIDFILE)
        except Exception as e:
            logging.warning(f"Could not remove PID file: {e}")

        logging.debug5(
            f"Cleaned up PIDs from {cls.PIDFILE}: attempted={attempted}, killed={killed}, failed={failed}"
        )

    @classmethod
    def clean_on_exit(cls) -> None:
        """Register cleanup to run automatically when the app exits."""
        if not cls._atexit_registered:
            atexit.register(cls._cleanup_pidfile)
            cls._atexit_registered = True
            logging.debug("🧹 Registered atexit handler for PID cleanup.")
