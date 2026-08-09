import os
import cv2
import numpy as np
from tqdm import tqdm

from .visualization import draw_pose


def load_keypoints(npz_path):
    """
    Load keypoints from NPZ file.
    
    Args:
        npz_path: Path to NPZ file
    
    Returns:
        kpts: Keypoints array (T, 17, 2)
        style: Detected style ("raw" or "refined")
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
    
    Args:
        kpts: Keypoints array (T, 17, 2), may be normalized
        width: Frame width
        height: Frame height
    
    Returns:
        Denormalized keypoints (T, 17, 2)
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
    
    Args:
        video_path: Input video file path
        npz_path: NPZ file with pose data
        output_path: Output MP4 file path
        style: "raw" or "refined" (auto-detect if None)
        fps_override: Override FPS (use video FPS if None)
    
    Returns:
        Dictionary with rendering statistics
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


def get_output_path(npz_path, output_arg, project_root):
    """Generate output path based on input."""
    if output_arg:
        return output_arg
    
    npz_basename = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(project_root, "data", "outputs", "output_mp4", f"{npz_basename}_overlay.mp4")
