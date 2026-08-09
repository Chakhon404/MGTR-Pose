import cv2
import numpy as np
import time
from ..core.yolo_wrapper import YOLOWrapper
from ..utils.visualization import draw_skeleton, add_info_text
from ..config import DEFAULT_MODEL, DEFAULT_CONF


class PureYOLOLive:
    def __init__(self, model_path=DEFAULT_MODEL, conf=DEFAULT_CONF, device="cpu", camera_id=1):
        self.model = YOLOWrapper(model_path, conf, device)
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.fps_ema = 0.0
    
    def run(self):
        print("[INFO] Running PURE YOLO Live")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            t_start = time.time()
            kpts = self.model.predict(frame)
            if kpts is None:
                kpts = np.full((17, 2), np.nan)
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
