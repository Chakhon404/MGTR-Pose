import cv2
import numpy as np
import os


class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
    
    def get_grayscale_small(self, flow_scale):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, None, fx=flow_scale, fy=flow_scale)
        return frame, gray_small
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()


def resolve_video_path(video_name, project_root):
    if os.path.exists(video_name):
        return video_name
    return os.path.join(project_root, "data", "inputs", video_name)


def resolve_model_path(model_name, project_root):
    if os.path.exists(model_name):
        return model_name
    return os.path.join(project_root, "models", model_name)


def generate_output_path(video_name, prefix, output_dir):
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    return os.path.join(output_dir, f"{prefix}{base_name}.npz")


def ensure_output_dir(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
