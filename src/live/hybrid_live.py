import cv2
import numpy as np
import time
from ..core.yolo_wrapper import YOLOWrapper
from ..core.optical_flow import SparseOpticalFlowTracker
from ..utils.visualization import draw_skeleton, add_info_text
from ..utils.math_utils import interpolate_keypoints_linear
from ..utils.video_utils import resolve_model_path
from ..config import DEFAULT_MODEL, DEFAULT_CONF, FLOW_SCALE, MOTION_THRESH_IDLE, MAX_SKIP_IDLE, MAX_SKIP_ACTIVE, PROJECT_ROOT


class HybridLiveBuffer:
    def __init__(self, model_path=DEFAULT_MODEL, conf=DEFAULT_CONF, device="cpu", camera_id=1,
                 flow_scale=FLOW_SCALE, motion_thr=MOTION_THRESH_IDLE, idle_skip=MAX_SKIP_IDLE, active_skip=MAX_SKIP_ACTIVE):
        model_path = resolve_model_path(model_path, PROJECT_ROOT)
        self.model = YOLOWrapper(model_path, conf, device)
        self.flow_tracker = SparseOpticalFlowTracker(flow_scale)
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.flow_scale = flow_scale
        self.motion_thr = motion_thr
        self.idle_skip = idle_skip
        self.active_skip = active_skip
        self.buffer = []
        self.prev_kpts_px = None
        self.prev_gray_small = None
        self.prev_kpts_small = None
        self.skip_count = 0
        self.total_compute_time = 0.0
        self.fps_ema = 0.0
        self.is_first_fps = True
    
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
        print("[INFO] Running Hybrid Live Buffer")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            t0 = time.time()
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray_small = cv2.resize(curr_gray, None, fx=self.flow_scale, fy=self.flow_scale)
            motion_score = 999.0
            if self.prev_gray_small is not None and self.prev_kpts_small is not None:
                motion_score, curr_kpts_small = self.flow_tracker.compute(self.prev_gray_small, curr_gray_small, self.prev_kpts_small)
            else:
                curr_kpts_small = None
            current_max_skip = self.idle_skip if motion_score < self.motion_thr else self.active_skip
            run_yolo = (motion_score >= 999.0) or (self.skip_count >= current_max_skip) or (self.prev_kpts_px is None)
            self.buffer.append({'frame': frame.copy(), 'motion_score': motion_score, 'skip_count': self.skip_count, 'max_skip': current_max_skip})
            self.total_compute_time += (time.time() - t0)
            if not run_yolo:
                self.skip_count += 1
                if curr_kpts_small is not None:
                    self.prev_kpts_small = curr_kpts_small
                self.prev_gray_small = curr_gray_small
                continue
            t0_yolo = time.time()
            result = self.model.predict(frame)
            curr_kpts_px = self._extract_keypoints(result)
            if not np.isnan(curr_kpts_px).all():
                valid_mask = ~np.isnan(curr_kpts_px).any(axis=1)
                valid_pts = curr_kpts_px[valid_mask]
                if len(valid_pts) > 0:
                    self.prev_kpts_small = (valid_pts * self.flow_scale).astype(np.float32).reshape(-1, 1, 2)
                else:
                    self.prev_kpts_small = None
            else:
                self.prev_kpts_small = None
            self.total_compute_time += (time.time() - t0_yolo)
            t0_interp = time.time()
            N = len(self.buffer)
            for i, item in enumerate(self.buffer):
                ratio = (i + 1) / N
                interp_kpts = interpolate_keypoints_linear(self.prev_kpts_px if self.prev_kpts_px is not None else curr_kpts_px, curr_kpts_px, ratio)
                item['display_frame'] = draw_skeleton(item['frame'].copy(), interp_kpts, (0, 255, 255))
            self.total_compute_time += (time.time() - t0_interp)
            curr_fps = N / self.total_compute_time if self.total_compute_time > 0 else 0
            if self.is_first_fps:
                self.fps_ema = curr_fps
                self.is_first_fps = False
            else:
                self.fps_ema = 0.8 * self.fps_ema + 0.2 * curr_fps
            for i, item in enumerate(self.buffer):
                frame_to_show = item['display_frame']
                add_info_text(frame_to_show, "MG-Pose", (10, 30), (0, 255, 255))
                add_info_text(frame_to_show, f"FPS: {self.fps_ema:.1f}", (10, 60), (255, 255, 255))
                mode_text = "YOLO (Keyframe)" if i == N - 1 else f"Interpolate ({i+1}/{N-1})"
                add_info_text(frame_to_show, f"Mode: {mode_text}", (10, 90), (0, 255, 0))
                cv2.imshow("Live: Hybrid Buffer", frame_to_show)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
            self.prev_kpts_px = curr_kpts_px.copy()
            self.prev_gray_small = curr_gray_small
            self.buffer.clear()
            self.skip_count = 0
            self.total_compute_time = 0.0
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--motion_thr", type=float, default=MOTION_THRESH_IDLE)
    args = parser.parse_args()
    app = HybridLiveBuffer(args.model, args.conf, args.device, args.camera, motion_thr=args.motion_thr)
    app.run()


if __name__ == "__main__":
    main()
