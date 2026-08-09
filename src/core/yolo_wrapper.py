import numpy as np
from ultralytics import YOLO


class YOLOWrapper:
    def __init__(self, model_path, conf=0.25, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
    
    def predict(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            
            kpts = results[0].keypoints.xy[idx].cpu().numpy()
            kpts[np.all(kpts == 0, axis=1)] = np.nan
            
            results[0]._keypoints = kpts
            return results[0]
        
        return None
