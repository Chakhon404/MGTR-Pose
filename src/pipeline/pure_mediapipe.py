import os
import numpy as np
from .base import BasePipeline


class PureMediaPipePipeline(BasePipeline):
    """Pure MediaPipe baseline: runs MediaPipe Pose on every frame."""
    def __init__(self, model_path="models/pose_landmarker_lite.task", video_path=None, conf=0.25, device="cpu"):
        super().__init__(model_path, video_path, conf, device)
        self.current_kpts = np.full((33, 2), np.nan, dtype=np.float32)
        self.normalize_in_base = True

    def _process_frame(self, frame, frame_idx):
        kpts_px = self.model.predict(frame)
        self.current_kpts = kpts_px
        return self.current_kpts

    def save_results(self, output_path):
        results = self.get_results()
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        K = results["keypoints"]
        np.savez_compressed(
            output_path,
            imgname=np.array(results["frame_names"]),
            keypoints_2d=K.reshape(len(K), -1),
            kpts=K,
            video_fps=results["video_fps"],
            processing_fps=results["processing_fps"],
            elapsed_time=results["elapsed_time"],
            width=results["width"],
            height=results["height"]
        )
