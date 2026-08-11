import cv2
import numpy as np
import time
from ..core.yolo_wrapper import YOLOWrapper
from ..utils.visualization import draw_skeleton, add_info_text
from ..utils.video_utils import resolve_model_path
from ..config import DEFAULT_MODEL, DEFAULT_CONF, PROJECT_ROOT


class PureYOLOLive:
    def __init__(self, model_path=DEFAULT_MODEL, conf=DEFAULT_CONF, device="cpu", camera_id=1):
        model_path = resolve_model_path(model_path, PROJECT_ROOT)
        self.model = YOLOWrapper(model_path, conf, device)
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.fps_ema = 0.0
    
    def _extract_keypoints(self, result):
        if result is None:
            return np.full((17, 2), np.nan, dtype=np.float32)
        if hasattr(result, '_keypoints'):
            return result._keypoints.copy().astype(np.float32)
        if result.keypoints is not None and len(result.keypoints) > 0:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = np.argmax(areas)
            else:
                idx = 0
            kpts = result.keypoints.xy[idx].cpu().numpy().astype(np.float32)
            kpts[np.all(kpts == 0, axis=1)] = np.nan
            return kpts
        return np.full((17, 2), np.nan, dtype=np.float32)
    
    def run(self):
        print("[INFO] Running PURE YOLO Live")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            t_start = time.time()
            result = self.model.predict(frame)
            kpts = self._extract_keypoints(result)
            t_process = time.time() - t_start
            curr_fps = 1.0 / t_process if t_process > 0 else 0
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * curr_fps
            frame = draw_skeleton(frame, kpts, (0, 0, 255))
            add_info_text(frame, "PURE YOLO", (10, 30), (0, 0, 255))
            add_info_text(frame, f"FPS: {self.fps_ema:.1f}", (10, 60), (0, 0, 255))
            cv2.imshow("Live: Pure YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera", type=int, default=1)
    args = parser.parse_args()
    app = PureYOLOLive(args.model, args.conf, args.device, args.camera)
    app.run()


if __name__ == "__main__":
    main()
