import cv2
import numpy as np


class SparseOpticalFlowTracker:
    def __init__(self, flow_scale=0.25, winSize=(15, 15), maxLevel=2):
        self.flow_scale = flow_scale
        self.winSize = winSize
        self.maxLevel = maxLevel
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    
    def compute(self, prev_gray, curr_gray, prev_pts):
        if prev_pts is None or len(prev_pts) == 0:
            return 100.0, None
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
            winSize=self.winSize,
            maxLevel=self.maxLevel,
            criteria=self.criteria
        )
        good_new = curr_pts[status == 1]
        good_old = prev_pts[status == 1]
        if len(good_new) == 0:
            return 100.0, None
        distances = np.linalg.norm(good_new - good_old, axis=1)
        motion_score = np.max(distances)
        return motion_score, curr_pts
    
    def prepare_keypoints(self, kpts_px, scale=None):
        if kpts_px is None or np.isnan(kpts_px).all():
            return None
        valid_mask = ~np.isnan(kpts_px).any(axis=1)
        valid_pts = kpts_px[valid_mask]
        if len(valid_pts) == 0:
            return None
        pts = valid_pts
        if scale is not None:
            pts = (valid_pts * scale).astype(np.float32)
        return pts.reshape(-1, 1, 2)


class DenseOpticalFlowCalculator:
    def __init__(self, flow_scale=0.25):
        self.flow_scale = flow_scale
    
    def compute(self, prev_gray_small, curr_gray_small):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_small, curr_gray_small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow
    
    def compute_motion_score(self, flow):
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(mag)
