from dataclasses import dataclass
from typing import Any
from queue import Queue

@dataclass
class SWMR_StorerJob:
    dataset_path: str
    image_data: Any
    attributes: dict
    result_queue: Queue
    attribute_process: str = "att_replace"  # optional
    handler_method: str = "store_image"  # or "handle_dataset"

