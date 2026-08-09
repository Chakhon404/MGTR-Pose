import cv2
import numpy as np


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

COCO_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 5), (0, 6)
]


def draw_skeleton(frame, kpts, color):
    if kpts is None or np.isnan(kpts).all():
        return frame
    for p in kpts:
        if not np.isnan(p[0]) and not np.isnan(p[1]):
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1)
    for edge in SKELETON:
        p1, p2 = kpts[edge[0]], kpts[edge[1]]
        if not np.isnan(p1).any() and not np.isnan(p2).any():
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
    return frame


def draw_pose(img, kpts_xy, style="refined"):
    """
    Draw pose skeleton with configurable style.
    
    Args:
        img: Input image frame
        kpts_xy: Keypoints array (17, 2) in pixel coordinates
        style: "raw" (green/yellow for Pure YOLO) or "refined" (purple/blue for Hybrid)
    
    Returns:
        Image with drawn pose
    """
    if kpts_xy is None or np.isnan(kpts_xy).all():
        return img
    
    if style == "raw":
        pt_color = (0, 255, 0)
        ln_color = (0, 255, 255)
    elif style == "refined":
        pt_color = (255, 0, 255)
        ln_color = (255, 0, 0)
    else:
        pt_color = (255, 0, 255)
        ln_color = (255, 0, 0)
    
    for i, j in COCO_EDGES:
        x1, y1 = kpts_xy[i]
        x2, y2 = kpts_xy[j]
        if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), ln_color, 2, cv2.LINE_AA)
    
    for x, y in kpts_xy:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(img, (int(x), int(y)), 3, pt_color, -1, cv2.LINE_AA)
    
    return img


def add_info_text(frame, text, position=(10, 30), color=(255, 255, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def create_display_frame(frame, kpts, fps, mode):
    display = draw_skeleton(frame.copy(), kpts, (0, 255, 255))
    add_info_text(display, "MG-Pose", (10, 30), (0, 255, 255))
    add_info_text(display, f"FPS: {fps:.1f}", (10, 60), (255, 255, 255))
    add_info_text(display, f"Mode: {mode}", (10, 90), (0, 255, 0))
    return display
