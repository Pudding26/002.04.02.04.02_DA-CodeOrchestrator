from queue import Queue

# Shared memory-safe queue used by all producers
store_queue = Queue(maxsize=1000)
