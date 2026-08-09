import cv2
import numpy as np
import time
import argparse
import os
import sys

from ..core.yolo_wrapper import YOLOWrapper
from ..core.optical_flow import SparseOpticalFlowTracker
from ..utils.visualization import draw_skeleton
from ..config import (
    DEFAULT_MODEL, DEFAULT_CONF, DATA_INPUTS_DIR, MODEL_DIR,
    FLOW_SCALE, MULTI_STREAM_IDLE_SKIP, MULTI_STREAM_ACTIVE_SKIP,
    MULTI_STREAM_MOTION_THR, MULTI_STREAM_DEFAULT_VIDEOS
)


def resolve_video_path(video_name):
    if os.path.exists(video_name):
        return video_name
    path = os.path.join(DATA_INPUTS_DIR, video_name)
    if os.path.exists(path):
        return path
    return video_name


def create_stream_state():
    return {
        'prev_gray_small': None,
        'prev_kpts_small': None,
        'curr_kpts_px': np.full((17, 2), np.nan),
        'skip_count': 0,
        'last_motion_score': 0.0
    }


def process_pure_stream(model, frame, state):
    res = model.predict(frame)
    kpts_px = np.full((17, 2), np.nan)
    
    if res is not None and res.keypoints is not None:
        if res.boxes is not None and len(res.boxes) > 0:
            idx = int(((res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) * 
                       (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1])).argmax())
        else:
            idx = 0
        kpts_px = res.keypoints.xy[idx].cpu().numpy()
        kpts_px[np.all(kpts_px == 0, axis=1)] = np.nan
    
    state['curr_kpts_px'] = kpts_px.copy()
    return "PURE YOLO", (0, 0, 255)


def process_hybrid_stream(model, frame, state, flow_tracker, flow_scale, motion_thr, idle_skip, active_skip):
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_gray_small = cv2.resize(curr_gray, None, fx=flow_scale, fy=flow_scale)
    motion_score = 999.0
    
    if state['prev_gray_small'] is not None and state['prev_kpts_small'] is not None:
        motion_score, curr_kpts_small = flow_tracker.compute(
            state['prev_gray_small'], curr_gray_small, state['prev_kpts_small']
        )
    else:
        curr_kpts_small = None
    
    if motion_score != 999.0:
        state['last_motion_score'] = motion_score
    
    if motion_score >= 999.0:
        current_max_skip = active_skip
    else:
        T_dyn = motion_thr * (flow_scale / 0.25)
        current_max_skip = idle_skip if motion_score < T_dyn else active_skip
    
    run_yolo = (motion_score >= 999.0) or (state['skip_count'] >= current_max_skip)
    
    if run_yolo:
        res = model.predict(frame)
        kpts_px = np.full((17, 2), np.nan)
        
        if res is not None and res.keypoints is not None:
            if res.boxes is not None and len(res.boxes) > 0:
                idx = int(((res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) * 
                           (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1])).argmax())
            else:
                idx = 0
            kpts_px = res.keypoints.xy[idx].cpu().numpy()
            kpts_px[np.all(kpts_px == 0, axis=1)] = np.nan
            
            valid_mask = ~np.isnan(kpts_px).any(axis=1)
            valid_pts = kpts_px[valid_mask]
            
            if len(valid_pts) >= 5:
                state['prev_kpts_small'] = (valid_pts * flow_scale).astype(np.float32).reshape(-1, 1, 2)
            else:
                state['prev_kpts_small'] = None
        else:
            state['prev_kpts_small'] = None
        
        state['curr_kpts_px'] = kpts_px.copy()
        state['skip_count'] = 0
        mode_text = "YOLO (Keyframe)"
        color = (0, 0, 255)
    else:
        state['skip_count'] += 1
        if curr_kpts_small is not None and len(curr_kpts_small) >= 5:
            state['prev_kpts_small'] = curr_kpts_small
        
        mode_text = f"SKIP ({state['skip_count']}/{current_max_skip}) Hold"
        color = (0, 255, 0)
    
    state['prev_gray_small'] = curr_gray_small
    
    return mode_text, color


def main():
    parser = argparse.ArgumentParser(description="Multi-Stream Stress Test")
    parser.add_argument("--mode", type=str, default="HYBRID", choices=["PURE", "HYBRID"], 
                        help="Processing mode: PURE or HYBRID")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="List of video files (default: 3 default videos)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="YOLO model file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Confidence threshold")
    parser.add_argument("--idle_skip", type=int, default=MULTI_STREAM_IDLE_SKIP,
                        help="Max frames to skip when idle")
    parser.add_argument("--active_skip", type=int, default=MULTI_STREAM_ACTIVE_SKIP,
                        help="Max frames to skip when active")
    parser.add_argument("--motion_thr", type=float, default=MULTI_STREAM_MOTION_THR,
                        help="Motion threshold")
    parser.add_argument("--flow_scale", type=float, default=FLOW_SCALE,
                        help="Flow resize factor")
    parser.add_argument("--display_width", type=int, default=640,
                        help="Display frame width")
    parser.add_argument("--display_height", type=int, default=480,
                        help="Display frame height")
    args = parser.parse_args()
    
    video_names = args.videos if args.videos else MULTI_STREAM_DEFAULT_VIDEOS
    
    video_paths = []
    for vn in video_names:
        vp = resolve_video_path(vn)
        if not os.path.exists(vp):
            print(f"[WARNING] Video not found: {vp}")
            continue
        video_paths.append(vp)
    
    if len(video_paths) == 0:
        print("[ERROR] No valid videos found!")
        return
    
    num_streams = len(video_paths)
    print(f"[INFO] Multi-Stream Stress Test")
    print(f"       Mode: {args.mode}")
    print(f"       Streams: {num_streams}")
    print(f"       Videos: {video_paths}")
    
    model_path = args.model
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, args.model)
    
    print(f"[INFO] Loading YOLO Model: {model_path}")
    model = YOLOWrapper(model_path, args.conf, "cpu")
    
    flow_tracker = SparseOpticalFlowTracker(args.flow_scale)
    
    caps = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {vp}")
        else:
            caps.append(cap)
    
    if len(caps) == 0:
        print("[ERROR] No videos could be opened!")
        return
    
    num_streams = len(caps)
    streams_state = [create_stream_state() for _ in range(num_streams)]
    
    refresh_ema = 0.0
    throughput_ema = 0.0
    is_first_fps = True
    
    print(f"[INFO] Starting Multi-Stream Test ({num_streams} streams)")
    print(f"       Press 'q' to quit")
    
    while True:
        t_start_loop = time.time()
        
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (args.display_width, args.display_height))
            else:
                frame = np.zeros((args.display_height, args.display_width, 3), dtype=np.uint8)
            frames.append(frame)
        
        if len(frames) < num_streams:
            break
        
        display_frames = []
        total_compute_time = 0.0
        
        for i in range(num_streams):
            frame = frames[i]
            state = streams_state[i]
            
            t0 = time.time()
            
            if args.mode == "PURE":
                mode_text, color = process_pure_stream(model, frame, state)
            else:
                mode_text, color = process_hybrid_stream(
                    model, frame, state, flow_tracker,
                    args.flow_scale, args.motion_thr,
                    args.idle_skip, args.active_skip
                )
            
            total_compute_time += (time.time() - t0)
            
            out_frame = draw_skeleton(frame.copy(), state['curr_kpts_px'], color)
            cv2.putText(out_frame, f"Cam {i+1} : {args.mode}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(out_frame, f"State: {mode_text}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if args.mode == "HYBRID":
                cv2.putText(out_frame, f"Motion: {state['last_motion_score']:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            display_frames.append(out_frame)
        
        loop_time = time.time() - t_start_loop
        
        curr_refresh_fps = 1.0 / loop_time if loop_time > 0 else 0
        curr_throughput_fps = num_streams / total_compute_time if total_compute_time > 0 else 0
        
        if is_first_fps:
            refresh_ema = curr_refresh_fps
            throughput_ema = curr_throughput_fps
            is_first_fps = False
        else:
            refresh_ema = (0.8 * refresh_ema) + (0.2 * curr_refresh_fps)
            throughput_ema = (0.8 * throughput_ema) + (0.2 * curr_throughput_fps)
        
        combined_grid = np.hstack(display_frames)
        bottom_bar = np.zeros((80, combined_grid.shape[1], 3), dtype=np.uint8)
        
        sys_color = (0, 255, 0) if refresh_ema > 15 else (0, 165, 255) if refresh_ema > 10 else (0, 0, 255)
        
        cv2.putText(bottom_bar, f"PER-CAMERA REFRESH RATE: {refresh_ema:.1f} FPS", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sys_color, 2)
        cv2.putText(bottom_bar, f"SYSTEM THROUGHPUT: {throughput_ema:.1f} Frames/Sec | Mode: {args.mode}", 
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        final_output = np.vstack((combined_grid, bottom_bar))
        
        cv2.imshow("Multi-Stream Stress Test", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Multi-Stream Test ended")


if __name__ == "__main__":
    main()
