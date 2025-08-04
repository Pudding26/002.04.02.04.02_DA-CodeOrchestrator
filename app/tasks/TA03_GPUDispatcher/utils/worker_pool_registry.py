import os
from dotenv import load_dotenv

load_dotenv()

BASE_DOMAIN = os.getenv("WORKER_BASE_DOMAIN", "leandrohome.de")

def create_worker_pool_registry(local_workers=None, runpod_workers=None):
    """
    Dynamically build a worker pool registry with subdomain-per-worker.
    
    Args:
        local_workers (list[int]): local worker indices
        runpod_workers (list[int]): runpod worker indices
    
    Returns:
        dict: worker_pool_registry
    """
    local_workers = local_workers or []
    runpod_workers = runpod_workers or []

    def make_local_worker(n: int):
        name = f"gpu_worker_{n}"
        return name, {
            "type": "local",
            "endpoint": None,   # local workers don't need Runpod API ID
            "auto_start": False,
            "cloudflare_hostname": f"https://gpu-worker-{n}.{BASE_DOMAIN}"
        }

    def make_runpod_worker(n: int):
        name = f"gpu_worker_{n}"
        return name, {
            "type": "runpod",
            "endpoint": os.getenv(f"GPU_WORKER_{n}_ENDPOINT"),  # Runpod API ID
            "auto_start": True,
            "cloudflare_hostname": f"https://gpu-worker-{n}.{BASE_DOMAIN}"
        }

    worker_pool_registry = {}

    for n in local_workers:
        key, entry = make_local_worker(n)
        worker_pool_registry[key] = entry

    for n in runpod_workers:
        key, entry = make_runpod_worker(n)
        worker_pool_registry[key] = entry

    return worker_pool_registry
