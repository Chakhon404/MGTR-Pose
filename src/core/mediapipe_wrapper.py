import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipePoseWrapper:
    """Wrapper for MediaPipe PoseLandmarker (33 keypoints)."""
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model task not found at: {model_path}")
        
        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def predict(self, frame_bgr):
        """
        Run pose inference on a BGR frame.
        
        Args:
            frame_bgr: OpenCV image (H, W, 3) in BGR format
            
        Returns:
            kpts_px: numpy array of shape (33, 2) in pixel coordinates,
                     or NaNs if no person detected.
        """
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]
            kpts = []
            for lm in landmarks:
                # lm.x and lm.y are normalized in [0, 1]
                px = float(lm.x * W)
                py = float(lm.y * H)
                kpts.append([px, py])
            return np.array(kpts, dtype=np.float32)
        else:
            return np.full((33, 2), np.nan, dtype=np.float32)
