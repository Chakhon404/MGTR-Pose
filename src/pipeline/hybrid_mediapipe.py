import os
import cv2
import numpy as np
from .base import BasePipeline
from ..core.optical_flow import SparseOpticalFlowTracker


class HybridMediaPipePipeline(BasePipeline):
    """Hybrid MediaPipe pipeline with Two-Gear Adaptive Skipping and Linear Interpolation (33 keypoints)."""
    def __init__(self, model_path="models/pose_landmarker_lite.task", video_path=None, conf=0.25, device="cpu",
                 flow_scale=0.25, motion_thr=5.0, max_skip=6, no_interp=False):
        super().__init__(model_path, video_path, conf, device)
        self.no_interp = no_interp
        self.flow_scale = flow_scale
        self.motion_thr = motion_thr
        self.max_skip = max_skip
        self.flow_tracker = SparseOpticalFlowTracker(flow_scale)
        
        self.prev_gray_small = None
        self.prev_kpts_px = None
        self.prev_kpts_small = None
        self.current_kpts = np.full((33, 2), np.nan, dtype=np.float32)
        
        self.skip_count = 0
        self.motion_score = 0.0
        self.current_max_skip = 0
        self.is_pure_mode = (max_skip == 0)
        self.normalize_in_base = True

    def _process_frame(self, frame, frame_idx):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not self.is_pure_mode:
            curr_gray_small = cv2.resize(curr_gray, None, fx=self.flow_scale, fy=self.flow_scale)
        else:
            curr_gray_small = None
            
        run_model = True
        motion_score = 0.0
        curr_kpts_small = None
        
        if not self.is_pure_mode and self.prev_gray_small is not None and self.prev_kpts_small is not None:
            motion_score, curr_kpts_small = self.flow_tracker.compute(
                self.prev_gray_small, curr_gray_small, self.prev_kpts_small
            )
            
            dynamic_threshold = self.motion_thr * (self.flow_scale / 0.25)
            
            if motion_score < dynamic_threshold:
                self.current_max_skip = self.max_skip  # Gear 1 (Slow)
            else:
                self.current_max_skip = 2              # Gear 2 (Fast)
                
            if motion_score < 999.0 and self.skip_count < self.current_max_skip:
                run_model = False
                self.prev_kpts_small = curr_kpts_small
            else:
                run_model = True
                
        if run_model:
            kpts = self.model.predict(frame)
            self.skip_count = 0
            
            if kpts is not None and not np.isnan(kpts).all():
                self.prev_kpts_px = kpts.copy()
                self.prev_kpts_small = (kpts * self.flow_scale).astype(np.float32).reshape(-1, 1, 2)
                self.current_kpts = kpts.copy()
            else:
                self.prev_kpts_px = np.full((33, 2), np.nan, dtype=np.float32)
                self.prev_kpts_small = None
                self.current_kpts = self.prev_kpts_px.copy()
        else:
            self.skip_count += 1
            self.prev_kpts_px = np.full((33, 2), np.nan, dtype=np.float32)
            self.current_kpts = np.full((33, 2), np.nan, dtype=np.float32)
            
        if self.prev_kpts_px is None:
            self.prev_kpts_px = np.full((33, 2), np.nan, dtype=np.float32)
            self.current_kpts = self.prev_kpts_px.copy()
            
        if not self.is_pure_mode:
            self.prev_gray_small = curr_gray_small
            
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
