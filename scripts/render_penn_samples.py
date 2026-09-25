import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.pipeline import (
    PureYOLOPipeline, HybridSparseFlowPipeline,
    PureMediaPipePipeline, HybridMediaPipePipeline
)
from src.utils.render import render_side_by_side

DEMO_OUT_DIR = "data/outputs/output_mp4"
TEMP_NPZ_DIR = "data/outputs/penn_action_results/output_npz"

os.makedirs(DEMO_OUT_DIR, exist_ok=True)
os.makedirs(TEMP_NPZ_DIR, exist_ok=True)

# 3 Sample Videos to Render
SAMPLES = [
    {
        "name": "jumping_jacks_1081",
        "video": "data/inputs/penn_action/penn_jumping_jacks_1081.mp4",
        "desc": "Jumping Jacks (Fast Motion / Outdoor Daylight)"
    },
    {
        "name": "squat_1676",
        "video": "data/inputs/penn_action/penn_squat_1676.mp4",
        "desc": "Squat (Medium Motion / Indoor Gym)"
    },
    {
        "name": "clean_and_jerk_0766",
        "video": "data/inputs/penn_action/penn_clean_and_jerk_0766.mp4",
        "desc": "Clean & Jerk (Full Body Weightlifting Platform)"
    }
]


def render_all_samples():
    print("=" * 80)
    print("      RENDERING SIDE-BY-SIDE SAMPLES (YOLO & MEDIAPIPE ON PENN ACTION)")
    print("=" * 80)
    
    generated_videos = []
    
    for s in SAMPLES:
        vpath = s["video"]
        vname = s["name"]
        print(f"\n>>> Processing Sample: {s['desc']} ({vpath})")
        
        # -----------------------------
        # 1. YOLOv8n (17 keypoints)
        # -----------------------------
        yolo_pure_npz = os.path.join(TEMP_NPZ_DIR, f"{vname}_yolo_pure.npz")
        yolo_hybr_npz = os.path.join(TEMP_NPZ_DIR, f"{vname}_yolo_hybrid.npz")
        
        print("  [1/4] Running Pure YOLOv8n...")
        p_yolo = PureYOLOPipeline(model_path="models/yolov8n-pose.pt", video_path=vpath, device="cpu")
        p_yolo.run()
        p_yolo.save_results(yolo_pure_npz)
        
        print("  [2/4] Running Hybrid YOLOv8n...")
        h_yolo = HybridSparseFlowPipeline(
            model_path="models/yolov8n-pose.pt", video_path=vpath, device="cpu",
            flow_scale=0.25, motion_thr=5.0, max_skip=6
        )
        h_yolo.run()
        h_yolo.save_results(yolo_hybr_npz)
        
        yolo_out_mp4 = os.path.join(DEMO_OUT_DIR, f"demo_penn_yolo_{vname}.mp4")
        print(f"  Rendering Side-by-Side: {yolo_out_mp4}")
        render_side_by_side(vpath, yolo_pure_npz, yolo_hybr_npz, yolo_out_mp4)
        generated_videos.append(yolo_out_mp4)
        
        # -----------------------------
        # 2. MediaPipe (33 keypoints)
        # -----------------------------
        mp_pure_npz = os.path.join(TEMP_NPZ_DIR, f"{vname}_mp_pure.npz")
        mp_hybr_npz = os.path.join(TEMP_NPZ_DIR, f"{vname}_mp_hybrid.npz")
        
        print("  [3/4] Running Pure MediaPipe (33 kpts)...")
        p_mp = PureMediaPipePipeline(model_path="models/pose_landmarker_lite.task", video_path=vpath, device="cpu")
        p_mp.run()
        p_mp.save_results(mp_pure_npz)
        
        print("  [4/4] Running Hybrid MediaPipe (33 kpts)...")
        h_mp = HybridMediaPipePipeline(
            model_path="models/pose_landmarker_lite.task", video_path=vpath, device="cpu",
            flow_scale=0.25, motion_thr=5.0, max_skip=6
        )
        h_mp.run()
        h_mp.save_results(mp_hybr_npz)
        
        mp_out_mp4 = os.path.join(DEMO_OUT_DIR, f"demo_penn_mediapipe_{vname}.mp4")
        print(f"  Rendering Side-by-Side: {mp_out_mp4}")
        render_side_by_side(vpath, mp_pure_npz, mp_hybr_npz, mp_out_mp4)
        generated_videos.append(mp_out_mp4)
        
    print("\n" + "=" * 80)
    print("      ALL SIDE-BY-SIDE DEMO VIDEOS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    for gv in generated_videos:
        print(f"  -> {gv}")


if __name__ == "__main__":
    render_all_samples()
