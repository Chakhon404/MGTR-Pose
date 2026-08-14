import os
import cv2
import numpy as np
from tqdm import tqdm

from .visualization import draw_pose


def load_keypoints(npz_path):
    """
    Load keypoints from NPZ file.
    """
    arr = np.load(npz_path, allow_pickle=True)
    
    if 'keypoints_2d' in arr:
        raw_data = arr['keypoints_2d']
        style = "refined"
    elif 'kpts' in arr:
        raw_data = arr['kpts']
        style = "raw"
    else:
        raise KeyError(f"Unknown keys in NPZ. Found: {list(arr.keys())}")
    
    if raw_data.ndim == 2 and raw_data.shape[1] == 34:
        kpts = raw_data.reshape(-1, 17, 2)
    else:
        kpts = raw_data
    
    return kpts.astype(np.float32), style


def denormalize_keypoints(kpts, width, height):
    """
    Denormalize keypoints from [0,1] to pixel coordinates.
    """
    kpts = kpts.copy()
    sample_val = np.nanmax(kpts)
    
    if sample_val < 2.0:
        kpts[..., 0] *= width
        kpts[..., 1] *= height
    
    return kpts


def render_video(video_path, npz_path, output_path, style=None, fps_override=None):
    """
    Render pose overlay on video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    
    kpts, detected_style = load_keypoints(npz_path)
    if style is None:
        style = detected_style
    
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps_override if fps_override else video_fps
    
    kpts = denormalize_keypoints(kpts, W, H)
    
    T = kpts.shape[0]
    num_frames = min(total_frames, T) if total_frames else T
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vout = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for t in tqdm(range(num_frames), ncols=100, desc="Rendering"):
        ok, frame = cap.read()
        if not ok:
            break
        
        k = kpts[t]
        frame = draw_pose(frame, k, style=style)
        vout.write(frame)
    
    cap.release()
    vout.release()
    
    return {
        "output_path": output_path,
        "frames_rendered": num_frames,
        "style": style,
        "fps": fps,
        "width": W,
        "height": H
    }


def render_side_by_side(video_path, npz_left, npz_right, output_path):
    if not os.path.exists(video_path): raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(npz_left): raise FileNotFoundError(f"NPZ Left not found: {npz_left}")
    if not os.path.exists(npz_right): raise FileNotFoundError(f"NPZ Right not found: {npz_right}")
    
    kpts_left, _ = load_keypoints(npz_left)
    kpts_right, _ = load_keypoints(npz_right)
    
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    kpts_left = denormalize_keypoints(kpts_left, W, H)
    kpts_right = denormalize_keypoints(kpts_right, W, H)
    
    num_frames = min(total_frames, kpts_left.shape[0], kpts_right.shape[0])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vout = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, H))
    
    for t in tqdm(range(num_frames), ncols=100, desc="Rendering"):
        ok, frame = cap.read()
        if not ok: break
        
        frame_left = frame.copy()
        frame_right = frame.copy()
        
        frame_left = draw_pose(frame_left, kpts_left[t], style="raw")
        frame_right = draw_pose(frame_right, kpts_right[t], style="refined")
        
        cv2.putText(frame_left, f"Frame: {t}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame_right, f"Frame: {t}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        combined = np.hstack((frame_left, frame_right))
        vout.write(combined)
    
    cap.release()
    vout.release()
    print(f"Saved side-by-side video to: {output_path}")
    return output_path


def get_output_path(npz_path, output_arg, project_root):
    """Generate output path based on input."""
    if output_arg:
        return output_arg
    
    npz_basename = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(project_root, "data", "outputs", "output_mp4", f"{npz_basename}_overlay.mp4")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render side-by-side video demo")
    parser.add_argument("--video", type=str, default="data/inputs/jumping_jack.mp4", help="Path to input video")
    parser.add_argument("--left", type=str, default="data/outputs/output_npz/ablation/abl6_model/pure_v8n_jumping_jack.npz", help="NPZ for left side")
    parser.add_argument("--right", type=str, default="data/outputs/output_npz/ablation/abl6_model/hybrid_v8n_jumping_jack.npz", help="NPZ for right side")
    parser.add_argument("--out", type=str, default="data/outputs/output_mp4/demo_jumping_jack_side_by_side.mp4", help="Output MP4 path")
    args = parser.parse_args()
    
    print(f"Generating side-by-side demo for {args.video}...")
    print(f"Left: {args.left}")
    print(f"Right: {args.right}")
    
    try:
        render_side_by_side(
            video_path=args.video,
            npz_left=args.left,
            npz_right=args.right,
            output_path=args.out
        )
    except Exception as e:
        print(f"Error generating demo: {e}")
