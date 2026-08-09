import numpy as np
import cv2


def interpolate_nans_inplace(x):
    T, J, D = x.shape
    for j in range(J):
        for d in range(D):
            s = x[:, j, d]
            mask = np.isnan(s)
            if mask.any() and (~mask).any():
                t = np.arange(T)
                s[mask] = np.interp(t[mask], t[~mask], s[~mask])
                x[:, j, d] = s


def interpolate_keypoints_linear(prev_kpts, curr_kpts, ratio):
    interp_kpts = np.full((17, 2), np.nan)
    for j in range(17):
        p1 = prev_kpts[j]
        p2 = curr_kpts[j]
        if not np.isnan(p1).any() and not np.isnan(p2).any():
            interp_kpts[j] = p1 + (p2 - p1) * ratio
        elif not np.isnan(p1).any():
            interp_kpts[j] = p1
        elif not np.isnan(p2).any():
            interp_kpts[j] = p2
    return interp_kpts


def normalize_keypoints(kpts, width, height):
    if kpts is None or np.isnan(kpts).all():
        return np.full((17, 2), np.nan)
    kp_norm = kpts.copy()
    kp_norm[:, 0] /= max(width, 1)
    kp_norm[:, 1] /= max(height, 1)
    return kp_norm


def select_largest_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def compute_motion_score_flow(flow):
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)


def create_empty_keypoints():
    return np.full((17, 2), np.nan, dtype=np.float32)
