import cv2
import numpy as np
from .base import BasePipeline
from ..core.optical_flow import SparseOpticalFlowTracker
from ..utils.math_utils import create_empty_keypoints


class HybridSparseFlowPipeline(BasePipeline):
    def __init__(self, model_path, video_path, conf=0.25, device="cpu",
                 flow_scale=0.25, motion_thr=3.0, max_skip=6, no_interp=False):
        super().__init__(model_path, video_path, conf, device)
        self.no_interp = no_interp
        self.flow_scale = flow_scale
        self.motion_thr = motion_thr
        self.max_skip = max_skip
        self.flow_tracker = SparseOpticalFlowTracker(flow_scale)
        
        self.prev_gray_small = None
        self.prev_kpts_px = None
        self.prev_kpts_small = None
        self.current_kpts = create_empty_keypoints()
        
        self.skip_count = 0
        self.motion_score = 0.0
        self.current_max_skip = 0  # Two-Gear: tracks current gear level
        
        self.is_pure_mode = (max_skip == 0)
    
    def _process_frame(self, frame, frame_idx):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not self.is_pure_mode:
            curr_gray_small = cv2.resize(curr_gray, None, fx=self.flow_scale, fy=self.flow_scale)
        else:
            curr_gray_small = None
        
        run_yolo = True
        motion_score = 0.0
        curr_kpts_small = None
        
        if not self.is_pure_mode and self.prev_gray_small is not None and self.prev_kpts_small is not None:
            motion_score, curr_kpts_small = self.flow_tracker.compute(
                self.prev_gray_small, curr_gray_small, self.prev_kpts_small
            )
            
            # --- Two-Gear Adaptive Decision (Algorithm 2 in thesis) ---
            # Adjust threshold dynamically based on flow_scale
            dynamic_threshold = self.motion_thr * (self.flow_scale / 0.25)
            
            if motion_score < dynamic_threshold:
                self.current_max_skip = self.max_skip  # Gear 1 (Slow): skip up to max_skip
            else:
                self.current_max_skip = 2              # Gear 2 (Fast): skip up to 2
            
            # Skip YOLO if tracking is valid AND within current gear's limit
            if motion_score < 999.0 and self.skip_count < self.current_max_skip:
                run_yolo = False
                self.prev_kpts_small = curr_kpts_small
            else:
                run_yolo = True
        
        if run_yolo:
            res = self.model.predict(frame)
            self.skip_count = 0
            
            found = False
            if res is not None:
                r = res
                if r.keypoints is not None and len(r.keypoints) > 0:
                    if r.boxes is not None and len(r.boxes) > 0:
                        idx = int(((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])).argmax())
                    else:
                        idx = 0
                    kpts = r.keypoints.xy[idx].cpu().numpy().astype(np.float32)
                    self.prev_kpts_px = kpts.copy()
                    self.prev_kpts_small = (kpts * self.flow_scale).astype(np.float32).reshape(-1, 1, 2)
                    found = True
            
            if not found:
                self.prev_kpts_px = np.full((17, 2), np.nan, np.float32)
                self.prev_kpts_small = None
                self.current_kpts = self.prev_kpts_px.copy()
            else:
                self.current_kpts = self.prev_kpts_px.copy()
        else:
            self.skip_count += 1
            # Output NaN for skipped frames -> linearly interpolated in post-processing
            self.prev_kpts_px = np.full((17, 2), np.nan, np.float32)
            self.current_kpts = np.full((17, 2), np.nan, np.float32)
            # prev_kpts_small already updated in decision block above,
            # keeping the optical flow tracking chain alive for next frame
        
        if self.prev_kpts_px is None:
            self.prev_kpts_px = np.full((17, 2), np.nan, np.float32)
            self.current_kpts = self.prev_kpts_px.copy()
        
        if not self.is_pure_mode:
            self.prev_gray_small = curr_gray_small
        
        return self.current_kpts
