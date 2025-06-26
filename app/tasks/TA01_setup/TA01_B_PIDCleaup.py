
import os, signal
import logging

class TA01_B_PIDCleaup:

    @classmethod
    def cleanup_pids(cls) -> None:
        PIDFILE = '/tmp/my_orchestrator_pids'
        if not os.path.exists(PIDFILE):
            logging.debug2("No PID file found — nothing to clean up.")
            return

        attempted = 0
        killed = 0
        failed = 0

        with open(PIDFILE) as f:
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

        os.remove(PIDFILE)
        logging.debug5(
            f"Cleaned up PIDs from {PIDFILE}: attempted={attempted}, killed={killed}, failed={failed}"
        )