import logging
import time
import os
import requests
from uuid import uuid4
from queue import Queue, Empty
from collections import defaultdict
import pandas as pd
from datetime import datetime, timezone, timedelta


import statistics


from app.tasks.TaskBase import TaskBase
from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out
from app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out

from app.tasks.TA03_GPUDispatcher.utils.worker_pool_registry import create_worker_pool_registry
from app.tasks.TA03_GPUDispatcher.utils.worker_registry import worker_registry
from app.tasks.TA03_GPUDispatcher.utils.bulk_update_status import bulk_update_status

from app.utils.API.GPU_Dispatcher import WORKER_BLACKLIST


class TA03_0_GPUDispatcher(TaskBase):
    """
    GPU Dispatcher Task:
    --------------------
    Responsible for orchestrating GPU workers, fetching jobs from the database,
    assigning them to available workers, and starting new workers if necessary.

    - Maintains worker registry (live workers) and pool registry (possible workers).
    - Fetches READY jobs from database and manages dispatch queue.
    - Ensures workers are started via Runpod API when all current workers are busy.
    - Dispatches jobs to workers respecting max jobs per worker.
    - Logs summary of dispatch cycle including failures and per-worker stats.
    """

    def setup(self):
        """Initialize dispatcher settings and state variables."""
        self.num_gpu_workers = int(self.instructions.get("num_gpu_workers", 5))
        self.max_jobs_per_worker = int(self.instructions.get("max_jobs_per_worker", 5))
        self.worker_registry = worker_registry
        
        self.bulk_update_status_count = bulk_update_status()
        
        self.worker_pool_registry = create_worker_pool_registry(local_workers=self.instructions.get("local_workers", [0]),
                                                               runpod_workers=self.instructions.get("runpod_workers", []))

        self.runpod_api_key = os.getenv("RUNPOD_API_KEY")
        self.runpod_api_base = "https://api.runpod.ai/v2"
        self.debug_mode = None
        self.HEARTBEAT_TIMEOUT_SECONDS = int(self.instructions.get("heartbeat_timeout_seconds", 95))


        # Queue of job UUIDs waiting to be dispatched
        self.dispatch_queue = Queue()

        # Track jobs already dispatched (to avoid duplication)
        self.dispatched_jobs = set()

        # Track which workers have been triggered to start
        self.already_started_workers = set()
        self.currently_started_workers = set()
        self.BASE_SLEEP_TIME = 10

        self.hist_stats = pd.DataFrame(columns=[
            "worker_name", "total_processed", "total_inits",
            "average_uptime", "average_throughput"
        ])
        self.last_csv_flush = datetime.now(timezone.utc)


    def run(self):
        """
        Main event loop:
        - Ensures enough workers are running.
        - Fetches new jobs if queue is empty.
        - Dispatches jobs to available workers.
        - Logs a consolidated dispatch summary every cycle.
        """
        logging.info("🌀 GPUDispatcher starting.")
        backoff = 0
        while True:
            from app.utils.logger.loggingWrapper import LoggingHandler
            
            LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG-2")
            LoggingHandler(logging_level=LOGGING_LEVEL)
            if not hasattr(self, "start_time"):
                self.start_time = datetime.now(timezone.utc)


            started_workers = self.ensure_workers()

            # Fetch jobs if nothing pending in dispatch queue
            if self.dispatch_queue.empty():
                jobs_df = self.check_for_jobs("modeler")

                # Collect queued job UUIDs from all registered workers
                all_queued_ids = set()
                for worker in self.worker_registry.values():
                    all_queued_ids.update(worker.get("queued_ids", []))

                # Only schedule jobs that are not already queued
                new_jobs = [
                    job_uuid for job_uuid in jobs_df["job_uuid"].tolist()
                    if job_uuid not in all_queued_ids
                ]

                for job_uuid in new_jobs:
                    self.dispatch_queue.put(job_uuid)

                logging.info(
                    f"📦 Fetched {len(new_jobs)} new modeler jobs from database. "
                    f"Already Queued: {len(all_queued_ids)}"
                )

                if self.shutdown_one_worker():
                    logging.info("🔻 Scaled down one idle Runpod worker.")



            # Track dispatch stats for summary
            dispatched_count = 0
            failure_reasons = defaultdict(int)
            dispatches_per_worker = defaultdict(int)

            # Process all jobs currently in dispatch queue
            queue_size = self.dispatch_queue.qsize()
            logging.debug2(f"🔄 Dispatching jobs from queue (size={queue_size})...")
            for _ in range(queue_size):
                if _ % 25 == 0:
                    logging.debug2(f"🔄 Dispatching job {_}...")
                try:
                    job_uuid = self.dispatch_queue.get_nowait()
                except Empty:
                    break

                success, reason, worker = self.dispatch_job(job_uuid)
                if success:
                    dispatched_count += 1
                    self.dispatched_jobs.add(job_uuid)
                    dispatches_per_worker[worker] += 1
                else:
                    # Requeue job for retry later
                    self.dispatch_queue.put(job_uuid)
                    failure_reasons[reason] += 1

            # Consolidated cycle summary log
            summary_lines = [
                "📊 Dispatch summary:",
                f"   - Dispatched: {dispatched_count}",
                f"   - Remaining in queue: {self.dispatch_queue.qsize()}",
            ]
            if failure_reasons:
                summary_lines.append("   - Failures:")
                for reason, count in failure_reasons.items():
                    summary_lines.append(f"     • {reason}: {count}")
            if dispatches_per_worker:
                summary_lines.append("   - Dispatches per worker:")
                for worker, count in dispatches_per_worker.items():
                    summary_lines.append(f"     • {worker}: {count}")
            summary_lines.append(f"   - Workers started this cycle: {started_workers}")
            self.log_performance_stats()
            logging.info("\n".join(summary_lines))

            if dispatched_count == 0:
                logging.debug2("🔄 No jobs dispatched this cycle, checking again soon.")
                backoff =1
            else:
                backoff = 1


            # Sleep before next dispatch cycle
            time.sleep(min(self.BASE_SLEEP_TIME * backoff * backoff, 600))

    def _increment(self, count_ref):
        """Utility: increment a count reference (not widely used)."""
        count_ref += 1
        return count_ref

    def cleanup(self):
        """Cleanup resources when dispatcher shuts down."""
        logging.info("🧹 GPUDispatcher cleanup complete.")

    def ensure_workers(self):
        """
        Ensure that there are enough workers to handle jobs:
        - Checks current worker registry and pool.
        - Starts new Runpod workers if all registered workers are full.
        """
        started = 0
        logging.debug2("🔎 Checking current worker registry...")

        available_workers = [
            w for w in self.worker_registry.values()
            if w.get("allow_jobs", False)
        ]

        logging.debug2(f"🧾 Registered workers: {list(self.worker_registry.keys())}")
        logging.debug2(f"📋 Pool: {list(self.worker_pool_registry.keys())}")

        if self.debug_mode:
            logging.debug2("🛠️ DEBUG MODE: Skipping worker scaling logic.")
            return started

        # Determine if all active workers are at full capacity
        all_busy = all(
            len(w.get("queued_ids", [])) + len(w.get("running_ids", [])) >= self.max_jobs_per_worker
            for w in available_workers
        )


        # If all are busy, try to start new workers from the pool
        if all_busy:
            for name, info in self.worker_pool_registry.items():
                if name in self.worker_registry:
                    continue  # Already registered and alive
                if name in self.currently_started_workers:
                    continue  # Already being started

                if info.get("auto_start", False) and info.get("type") == "runpod":
                    logging.info(f"🟡 Starting worker {name} via endpoint {info.get('endpoint')}")
                    job_id = self.start_worker(info.get("endpoint"))
                    if job_id:
                        self.currently_started_workers.add(name)
                        started += 1
        else:
            logging.debug2("👌 Worker capacity is sufficient.")

        return started

    def start_worker(self, endpoint_id):
        """
        Trigger a Runpod serverless worker via its endpoint.
        Returns job_id if successful, None otherwise.
        """
        if not endpoint_id:
            logging.error("❌ Cannot start worker — endpoint ID is missing.")
            return None

        url = f"{self.runpod_api_base}/{endpoint_id}/run"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }

        payload = {"input": {"startup_mode": "worker_init"}}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.ok:
                job_id = response.json().get("id")
                logging.info(f"🚀 Triggered serverless worker {endpoint_id} (job_id={job_id})")
                return job_id
            else:
                logging.error(f"❌ Failed to start worker at {endpoint_id}: {response.text}")
        except Exception as e:
            logging.error(f"❌ Exception while starting worker: {e}")
        return None

    def dispatch_job(self, job_uuid):
        """
        Try to dispatch a job to an available worker.

        Uses both queued_ids and running_ids for capacity checks.
        """
        for worker_name, status_info in self.worker_registry.items():
            if not status_info.get("allow_jobs", True):
                continue

            # Capacity = queued + running
            queued_ids = set(status_info.get("queued_ids", []))
            running_ids = set(status_info.get("running_ids", []))
            active_jobs = len(queued_ids) + len(running_ids)

            if active_jobs >= self.max_jobs_per_worker:
                continue  # worker full

            pool_info = self.worker_pool_registry.get(worker_name.lower(), {})
            base_url = (
                pool_info.get("cloudflare_hostname")
            )
            url = f"{base_url}/start-job".replace("_", "-")

            payload = {"job_type": "modeler", "payload": {"job_uuids": [job_uuid]}}

            try:
                r = requests.post(url, json=payload, timeout=5)
                if r.ok:
                    # Keep local accounting optimistic until heartbeat confirms
                    status_info["queued_ids"] = set(status_info.get("queued_ids", []))
                    status_info["queued_ids"].add(job_uuid)
                    return True, None, worker_name
            except requests.exceptions.Timeout:
                logging.warning(f"⏳ Timeout while dispatching job {job_uuid} to {worker_name}, retrying...")
                continue
            except requests.exceptions.RequestException:
                logging.error(f"❌ RequestException while dispatching job {job_uuid} to {worker_name}.")
                continue

        return False, "no_available_worker", None



    def check_for_jobs(self, job_type):
        """
        Query DB for jobs of a given type that are READY to be scheduled.
        Returns a pandas DataFrame of jobs.
        """
        filter_model = FilterModel.from_human_filter({
            "contains": {"status": "READY", "job_type": job_type}
        })
        return WorkerJobs_Out.fetch(filter_model=filter_model, stream=False)

    def log_performance_stats(self):
        now = datetime.now(timezone.utc)

        snapshot_rows = []

        for worker_name, worker in self.worker_registry.items():
            finished = set(worker.get("finished_ids", []))
            successfull = set(worker.get("successfull_ids", []))

            processed = len(finished)
            active = len(worker.get("queued_ids", [])) + len(worker.get("running_ids", []))

            # normalize start_time
            worker_start_time = worker.get("start_time")
            if worker_start_time is None:
                worker_start_time = now
            elif isinstance(worker_start_time, str):
                worker_start_time = datetime.fromisoformat(worker_start_time)

            uptime_seconds = (now - worker_start_time).total_seconds()
            uptime_minutes = uptime_seconds / 60.0

            throughput = processed / uptime_minutes if uptime_minutes > 0 else 0.0
            success_rate = len(successfull) / (processed or 1)

            # Current run row
            snapshot_rows.append({
                "worker_name": worker_name,
                "start_time": worker_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "processed": processed,
                "active": active,
                "throughput": throughput,
                "success_rate": success_rate,
                "uptime_minutes": round(uptime_minutes, 2),
                "current_version": worker.get("current_version", "unknown"),
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            })

        snapshot_df = pd.DataFrame(snapshot_rows)

        # ---- Step 2: Update master history (deduplicated by worker_name + start_time) ----
        if not snapshot_df.empty:
            if self.hist_stats.empty:
                self.hist_stats = snapshot_df
            else:
                # Drop any existing rows for runs we are updating, then append
                keys = ["worker_name", "start_time"]
                self.hist_stats = (
                    pd.concat([
                        self.hist_stats[~self.hist_stats.set_index(keys).index.isin(
                            snapshot_df.set_index(keys).index
                        )],
                        snapshot_df
                    ])
                    .reset_index(drop=True)
                )

        # ---- Step 3: Enriched view (worker-level aggregates across *all runs*) ----
        agg_df = (
            self.hist_stats
            .groupby("worker_name", as_index=False)
            .agg(
                lt_proc_sum=("processed", "sum"),                # total processed across all runs
                lt_max_proc=("processed", "max"),         # max processed across runs
                total_inits=("start_time", "count"),                 # number of runs
                lt_avg_through=("throughput", "mean"),      # avg throughput across runs
                lt_avg_success=("success_rate", "mean"),  # avg success rate across runs
                lt_max_up=("uptime_minutes", "max"),       # max uptime seen
                lt_min_up=("uptime_minutes", "min"),       # min uptime seen
                lt_avg_up=("uptime_minutes", "mean"),       # avg uptime across runs
                idle_inits=("processed", lambda x: (x == 0).sum()),  # runs with 0 processed
            )
        )



        enriched_df = pd.merge(
            snapshot_df,
            agg_df,
            on="worker_name",
            how="left"
        )

        # ---- Step 4: Logging ----
        if not enriched_df.empty:
            logging.info("\n📊 Worker Stats (current + lifetime):\n" + enriched_df.to_string(index=False))

        # ---- Step 5: Persist ----
        if (now - self.last_csv_flush) > timedelta(minutes=5):
            self.hist_stats.to_csv("worker_stats.csv", index=False)
            self.last_csv_flush = now
            logging.info("💾 Worker stats persisted to worker_stats.csv")







    def purge_dead_workers(self):
        """Remove workers whose heartbeat timed out."""
        now = datetime.now(timezone.utc)
        dead = []

        for worker_name, info in list(worker_registry.items()):
            last_seen = info.get("last_seen")
            if not last_seen:
                continue

            # last_seen must be a datetime
            if isinstance(last_seen, str):
                try:
                    last_seen = datetime.fromisoformat(last_seen)
                except Exception:
                    continue

            if (now - last_seen).total_seconds() > self.HEARTBEAT_TIMEOUT_SECONDS:
                dead.append(worker_name)

        for name in dead:
            worker_registry.pop(name, None)
            logging.warning(f"💀 Worker {name} removed due to heartbeat timeout ({self.HEARTBEAT_TIMEOUT_SECONDS}s)")
        
        for name in WORKER_BLACKLIST:
            if name in worker_registry:
                worker_registry.pop(name, None)
                logging.warning(f"⛔ Worker {name} is blacklisted and has been removed")

    def shutdown_one_worker(self):
        """
        Shutdown the highest-numbered Runpod worker if:
        - The dispatch queue is empty
        - No worker has active jobs (queued or running)
        """
        # Check dispatch queue
        if not self.dispatch_queue.empty():
            return False

        # Check if any worker still has active jobs
        for worker_name, info in self.worker_registry.items():
            active_jobs = len(info.get("queued_ids", [])) + len(info.get("running_ids", []))
            if active_jobs > 0:
                return False  # Still work being processed, skip shutdown

        # Gather active runpod workers
        runpod_workers = []
        for name, info in self.worker_pool_registry.items():
            if info["type"] != "runpod":
                continue

            # normalize for comparison
            name_lc = name.lower()
            registry_lc = {k.lower(): k for k in self.worker_registry.keys()}

            if name_lc in registry_lc:
                # use the original-cased name from worker_registry for consistency
                orig_name = registry_lc[name_lc]
                runpod_workers.append((orig_name, info))
        if not runpod_workers:
            return False

        # Pick highest-numbered worker
        runpod_workers.sort(key=lambda x: int(x[0].split("_")[-1]), reverse=True)
        worker_name, info = runpod_workers[0]
        endpoint_id = info.get("endpoint")

        if not endpoint_id:
            logging.warning(f"⚠️ Worker {worker_name} has no endpoint, skipping shutdown.")
            return False

        # Call purge (shutdown)
        headers = {"Authorization": f"Bearer {self.runpod_api_key}"}
        purge_url = f"{self.runpod_api_base}/{endpoint_id}/purge-queue"

        try:
            r = requests.post(purge_url, headers=headers, timeout=10)
            if r.ok:
                logging.info(f"💤 Worker {worker_name} shut down successfully (purged).")
                self.worker_registry.pop(worker_name, None)
                return True
            else:
                logging.error(f"❌ Failed to purge {worker_name}: {r.status_code} {r.text}")
        except Exception as e:
            logging.error(f"❌ Exception shutting down {worker_name}: {e}")

        return False

