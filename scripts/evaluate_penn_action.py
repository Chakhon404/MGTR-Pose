import os
import sys
import glob

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pipeline import (
    PureYOLOPipeline, HybridSparseFlowPipeline,
    PureMediaPipePipeline, HybridMediaPipePipeline
)
from src.utils.math_utils import compute_motion_score_flow

INPUT_DIR = "data/inputs/penn_action"
OUTPUT_DIR = "data/outputs/penn_action_results"
OUTPUT_NPZ_DIR = os.path.join(OUTPUT_DIR, "output_npz")

# Penn Action joints (1..12) to COCO joints (5..16)
PENN_TO_COCO_JOINTS = [
    (1, 5),   # Left shoulder
    (2, 6),   # Right shoulder
    (3, 7),   # Left elbow
    (4, 8),   # Right elbow
    (5, 9),   # Left wrist
    (6, 10),  # Right wrist
    (7, 11),  # Left hip
    (8, 12),  # Right hip
    (9, 13),  # Left knee
    (10, 14), # Right knee
    (11, 15), # Left ankle
    (12, 16), # Right ankle
]

# Penn Action joints (1..12) to MediaPipe joints (11..16, 23..28)
PENN_TO_MP_JOINTS = [
    (1, 11),  # Left shoulder
    (2, 12),  # Right shoulder
    (3, 13),  # Left elbow
    (4, 14),  # Right elbow
    (5, 15),  # Left wrist
    (6, 16),  # Right wrist
    (7, 23),  # Left hip
    (8, 24),  # Right hip
    (9, 25),  # Left knee
    (10, 26), # Right knee
    (11, 27), # Left ankle
    (12, 28), # Right ankle
]


def calculate_jitter(kpts):
    """Calculate mean acceleration jitter across valid consecutive frames."""
    if len(kpts) < 3:
        return 0.0
    accels = kpts[2:] - 2 * kpts[1:-1] + kpts[:-2]
    mags = np.linalg.norm(accels, axis=-1)
    return float(np.nanmean(mags))


def calculate_mpjpe_to_gt(kpts_pred, gt_npz_path, model_type="yolo"):
    """
    Calculate MPJPE against 12 human-annotated ground truth body joints.
    Both pred and gt are normalized in [0, 1].
    """
    gt = np.load(gt_npz_path)
    gt_x = gt["x"]            # (T, 13)
    gt_y = gt["y"]            # (T, 13)
    gt_vis = gt["visibility"]  # (T, 13)
    W, H = float(gt["width"]), float(gt["height"])
    
    T = min(len(kpts_pred), len(gt_x))
    joint_mapping = PENN_TO_COCO_JOINTS if model_type == "yolo" else PENN_TO_MP_JOINTS
    
    errors = []
    for t in range(T):
        for penn_idx, pred_idx in joint_mapping:
            if gt_vis[t, penn_idx] > 0:
                gx = gt_x[t, penn_idx] / W
                gy = gt_y[t, penn_idx] / H
                px = kpts_pred[t, pred_idx, 0]
                py = kpts_pred[t, pred_idx, 1]
                
                if not (np.isnan(px) or np.isnan(py)):
                    dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                    errors.append(dist)
                    
    return float(np.mean(errors)) if len(errors) > 0 else 0.0


def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)
    
    video_files = sorted(glob.glob(os.path.join(INPUT_DIR, "penn_*.mp4")))
    print(f"Running Penn Action Benchmark on {len(video_files)} videos...\n")
    
    records = []
    
    for vpath in tqdm(video_files, desc="Evaluating videos"):
        vname = os.path.splitext(os.path.basename(vpath))[0]
        gt_path = os.path.join(INPUT_DIR, f"{vname}_gt.npz")
        
        # 1. Pure YOLOv8n
        p_yolo = PureYOLOPipeline(model_path="models/yolov8n-pose.pt", video_path=vpath, device="cpu")
        res_p_yolo = p_yolo.run()
        fps_p_yolo = res_p_yolo["processing_fps"]
        jit_p_yolo = calculate_jitter(res_p_yolo["keypoints"])
        mpjpe_p_yolo = calculate_mpjpe_to_gt(res_p_yolo["keypoints"], gt_path, "yolo")
        
        # 2. Hybrid YOLOv8n
        h_yolo = HybridSparseFlowPipeline(
            model_path="models/yolov8n-pose.pt", video_path=vpath, device="cpu",
            flow_scale=0.25, motion_thr=5.0, max_skip=6
        )
        res_h_yolo = h_yolo.run()
        fps_h_yolo = res_h_yolo["processing_fps"]
        jit_h_yolo = calculate_jitter(res_h_yolo["keypoints"])
        mpjpe_h_yolo = calculate_mpjpe_to_gt(res_h_yolo["keypoints"], gt_path, "yolo")
        
        # 3. Pure MediaPipe
        p_mp = PureMediaPipePipeline(model_path="models/pose_landmarker_lite.task", video_path=vpath, device="cpu")
        res_p_mp = p_mp.run()
        fps_p_mp = res_p_mp["processing_fps"]
        jit_p_mp = calculate_jitter(res_p_mp["keypoints"])
        mpjpe_p_mp = calculate_mpjpe_to_gt(res_p_mp["keypoints"], gt_path, "mediapipe")
        
        # 4. Hybrid MediaPipe
        h_mp = HybridMediaPipePipeline(
            model_path="models/pose_landmarker_lite.task", video_path=vpath, device="cpu",
            flow_scale=0.25, motion_thr=5.0, max_skip=6
        )
        res_h_mp = h_mp.run()
        fps_h_mp = res_h_mp["processing_fps"]
        jit_h_mp = calculate_jitter(res_h_mp["keypoints"])
        mpjpe_h_mp = calculate_mpjpe_to_gt(res_h_mp["keypoints"], gt_path, "mediapipe")
        
        records.append({
            "Video": vname,
            "YOLO_Pure_FPS": fps_p_yolo,
            "YOLO_Hybrid_FPS": fps_h_yolo,
            "YOLO_Speedup": fps_h_yolo / fps_p_yolo if fps_p_yolo > 0 else 0,
            "YOLO_Pure_Jitter": jit_p_yolo,
            "YOLO_Hybrid_Jitter": jit_h_yolo,
            "YOLO_Jitter_Reduction_%": (jit_p_yolo - jit_h_yolo) / jit_p_yolo * 100 if jit_p_yolo > 0 else 0,
            "YOLO_Pure_MPJPE": mpjpe_p_yolo,
            "YOLO_Hybrid_MPJPE": mpjpe_h_yolo,
            
            "MP_Pure_FPS": fps_p_mp,
            "MP_Hybrid_FPS": fps_h_mp,
            "MP_Speedup": fps_h_mp / fps_p_mp if fps_p_mp > 0 else 0,
            "MP_Pure_Jitter": jit_p_mp,
            "MP_Hybrid_Jitter": jit_h_mp,
            "MP_Jitter_Reduction_%": (jit_p_mp - jit_h_mp) / jit_p_mp * 100 if jit_p_mp > 0 else 0,
            "MP_Pure_MPJPE": mpjpe_p_mp,
            "MP_Hybrid_MPJPE": mpjpe_h_mp,
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "penn_action_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to: {csv_path}\n")
    
    # Print clean summary
    print("=" * 80)
    print("        PENN ACTION DATASET BENCHMARK RESULTS (AVERAGES OVER 12 CLIPS)")
    print("=" * 80)
    print(f"YOLOv8n Pure FPS:          {df['YOLO_Pure_FPS'].mean():.2f} FPS")
    print(f"YOLOv8n Hybrid FPS:        {df['YOLO_Hybrid_FPS'].mean():.2f} FPS  --> Speedup: {df['YOLO_Speedup'].mean():.2f}x")
    print(f"YOLOv8n Jitter Reduction:  {df['YOLO_Jitter_Reduction_%'].mean():.1f}%  (Pure: {df['YOLO_Pure_Jitter'].mean():.5f} -> Hybrid: {df['YOLO_Hybrid_Jitter'].mean():.5f})")
    print(f"YOLOv8n MPJPE (vs Real GT): Pure {df['YOLO_Pure_MPJPE'].mean():.5f} -> Hybrid {df['YOLO_Hybrid_MPJPE'].mean():.5f}")
    print("-" * 80)
    print(f"MediaPipe Pure FPS:        {df['MP_Pure_FPS'].mean():.2f} FPS")
    print(f"MediaPipe Hybrid FPS:      {df['MP_Hybrid_FPS'].mean():.2f} FPS  --> Speedup: {df['MP_Speedup'].mean():.2f}x")
    print(f"MediaPipe Jitter Reduction:{df['MP_Jitter_Reduction_%'].mean():.1f}%  (Pure: {df['MP_Pure_Jitter'].mean():.5f} -> Hybrid: {df['MP_Hybrid_Jitter'].mean():.5f})")
    print(f"MediaPipe MPJPE (vs Real GT): Pure {df['MP_Pure_MPJPE'].mean():.5f} -> Hybrid {df['MP_Hybrid_MPJPE'].mean():.5f}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
