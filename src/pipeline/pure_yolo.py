import os
import numpy as np
from .base import BasePipeline
from ..utils.math_utils import create_empty_keypoints


class PureYOLOPipeline(BasePipeline):
    def __init__(self, model_path, video_path, conf=0.25, device="cpu"):
        super().__init__(model_path, video_path, conf, device)
        self.current_kpts = np.full((17, 2), np.nan, dtype=np.float32)
        self.normalize_in_base = False
    
    def _process_frame(self, frame, frame_idx):
        result = self.model.predict(frame)
        
        current_kpts = np.full((17, 2), np.nan, dtype=np.float32)
        
        if result is not None and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            
            kpts = result.keypoints.xy[idx].cpu().numpy()
            kpts[np.all(kpts == 0, axis=1)] = np.nan
            
            kpts[:, 0] /= max(self.width, 1)
            kpts[:, 1] /= max(self.height, 1)
            
            current_kpts = kpts
        
        self.current_kpts = current_kpts
        return self.current_kpts
    
    def save_results(self, output_path):
        results = self.get_results()
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(
            output_path,
            kpts=results["keypoints"],
            video_fps=results["video_fps"],
            processing_fps=results["processing_fps"],
            elapsed_time=results["elapsed_time"],
            total_frames=len(results["keypoints"])
        )
