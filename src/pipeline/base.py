import os
import time
import numpy as np
import cv2
from abc import ABC, abstractmethod

from ..utils.math_utils import interpolate_nans_inplace, create_empty_keypoints


class BasePipeline(ABC):
    def __init__(self, model_path, video_path, conf=0.25, device="cpu"):
        self.model_path = model_path
        self.video_path = video_path
        self.conf = conf
        self.device = device
        self.cap = None
        self.model = None
        self.width = 0
        self.height = 0
        self.video_fps = 0
        self.total_frames = 0
        self.keypoints_all = []
        self.frame_names = []
        self.start_time = None
        self.processing_fps = 0
        self.elapsed_time = 0
        self.normalize_in_base = True
    
    @abstractmethod
    def _process_frame(self, frame, frame_idx):
        pass
    
    def initialize(self):
        from ..core.yolo_wrapper import YOLOWrapper
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.model = YOLOWrapper(self.model_path, self.conf, self.device)
    
    def run(self):
        self.initialize()
        self.start_time = time.time()
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            current_kpts = self._process_frame(frame, frame_idx)
            
            if self.normalize_in_base:
                kp_norm = current_kpts.copy()
                kp_norm[:, 0] /= max(self.width, 1)
                kp_norm[:, 1] /= max(self.height, 1)
                self.keypoints_all.append(kp_norm.reshape(-1))
            else:
                self.keypoints_all.append(current_kpts.reshape(-1))
            
            self.frame_names.append(f"{frame_idx:06d}.jpg")
            frame_idx += 1
        self._finalize()
        return self.get_results()
    
    def _finalize(self):
        if self.cap is not None:
            self.cap.release()
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        self.processing_fps = len(self.keypoints_all) / self.elapsed_time if self.elapsed_time > 0 else 0
    
    def get_results(self):
        K = np.stack(self.keypoints_all, axis=0).reshape(-1, 17, 2).astype("float32")
        interpolate_nans_inplace(K)
        return {
            "keypoints": K,
            "frame_names": self.frame_names,
            "video_fps": self.video_fps,
            "processing_fps": self.processing_fps,
            "elapsed_time": self.elapsed_time,
            "width": self.width,
            "height": self.height
        }
    
    def save_results(self, output_path):
        results = self.get_results()
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(
            output_path,
            imgname=np.array(results["frame_names"]),
            keypoints_2d=results["keypoints"].reshape(len(results["keypoints"]), 34),
            video_fps=results["video_fps"],
            processing_fps=results["processing_fps"],
            elapsed_time=results["elapsed_time"],
            width=results["width"],
            height=results["height"]
        )
