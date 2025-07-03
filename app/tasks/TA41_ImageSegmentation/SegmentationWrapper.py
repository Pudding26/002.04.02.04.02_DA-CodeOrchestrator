import numpy as np

from app.tasks.TA41_ImageSegmentation.Preprocessor import Preprocessor
from app.tasks.TA41_ImageSegmentation.Segmenter import Segmenter
from app.tasks.TA41_ImageSegmentation.FeatureExtractor import FeatureExtractor
import time

class SegmentationWrapper:
    def __init__(self, config: dict, gpu_mode = False):
        self.preprocessor = Preprocessor(config)
        self.segmenter = Segmenter(config)
        self.extractor = FeatureExtractor()
        self.gpu_mode = gpu_mode

    def run_single(self, image: np.ndarray) -> dict:
        start_time = time.time()
        
        filtered, new_gray = self.preprocessor.apply_one(image)
        after_preprocessing_time = time.time()

        mask = self.segmenter.apply_one(filtered)
        after_segmenter_time = time.time()

        if self.gpu_mode == False: # Allows to combine the extraction either with or without GPU
            features = self.extractor.apply_one(mask)
            after_extractor_time = time.time()
            
            stats = {
                "preprocessing_time": after_preprocessing_time - start_time,
                "segmentation_time": after_segmenter_time - after_preprocessing_time,
                "feature_extraction_time": after_extractor_time - after_segmenter_time,
                "total_time": after_extractor_time - start_time
            }
            
            result = {
                "filtered_image": filtered, 
                "new_gray": new_gray, 
                "segmentation_mask": mask, 
                "features": features,
                "stats" : stats                    
            }
            


            return result
        
        else:
            return {
                "filtered_image": filtered, 
                "new_gray": new_gray, 
                "segmentation_mask": mask, 
                "features": None
            }
        