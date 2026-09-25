import os
import glob
import cv2
import scipy.io
import numpy as np
from tqdm import tqdm

PENN_ROOT = "C:/RM/penn-action/Penn_Action"
OUTPUT_DIR = "data/inputs/penn_action"

# 12 Curated Representative Sequences (Single-person, Front View, 100% visibility, Official Test Split)
SELECTED_SEQS = [
    ("1081", "jumping_jacks", "Fast / Dynamic Motion (Outdoor)"),
    ("1124", "jumping_jacks", "Fast / Dynamic Motion (Gym/Indoor)"),
    ("1043", "jumping_jacks", "Fast / Dynamic Motion (Studio)"),
    ("1068", "jumping_jacks", "Fast / Dynamic Motion (Home Indoor)"),
    ("1676", "squat", "Medium / Lower Body Motion (Gym)"),
    ("1681", "squat", "Medium / Lower Body Motion (Female Subject)"),
    ("1700", "squat", "Medium / Lower Body Motion (Male Subject)"),
    ("1785", "squat", "Medium / Lower Body Motion (Daylight)"),
    ("1301", "pullup", "Upper Body Compound Motion"),
    ("1201", "pullup", "Upper Body Compound Motion (High Bar)"),
    ("0766", "clean_and_jerk", "Full Body Olympic Weightlifting"),
    ("0759", "clean_and_jerk", "Full Body Weightlifting (Platform)")
]


def curate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []
    
    print(f"Curating {len(SELECTED_SEQS)} representative Penn Action sequences into '{OUTPUT_DIR}'...\n")
    
    for seq_id, action, description in tqdm(SELECTED_SEQS, desc="Processing sequences"):
        label_file = os.path.join(PENN_ROOT, "labels", f"{seq_id}.mat")
        frames_dir = os.path.join(PENN_ROOT, "frames", seq_id)
        
        if not os.path.exists(label_file) or not os.path.exists(frames_dir):
            print(f"[WARN] Skipping {seq_id}: files not found")
            continue
            
        mat = scipy.io.loadmat(label_file)
        x_coords = mat["x"]            # (nframes, 13)
        y_coords = mat["y"]            # (nframes, 13)
        visibility = mat["visibility"]  # (nframes, 13)
        dims = mat["dimensions"][0]    # [H, W, nframes]
        H, W, nframes = int(dims[0]), int(dims[1]), int(dims[2])
        
        # Read frames and create MP4
        frame_pattern = os.path.join(frames_dir, "*.jpg")
        frame_files = sorted(glob.glob(frame_pattern))
        
        if len(frame_files) == 0:
            print(f"[WARN] No image frames in {frames_dir}")
            continue
            
        out_video_name = f"penn_{action}_{seq_id}.mp4"
        out_video_path = os.path.join(OUTPUT_DIR, out_video_name)
        out_gt_name = f"penn_{action}_{seq_id}_gt.npz"
        out_gt_path = os.path.join(OUTPUT_DIR, out_gt_name)
        
        # Write MP4 video at 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(out_video_path, fourcc, 30.0, (W, H))
        
        for fpath in frame_files:
            img = cv2.imread(fpath)
            if img is not None:
                # Ensure correct dimension
                if img.shape[0] != H or img.shape[1] != W:
                    img = cv2.resize(img, (W, H))
                vout.write(img)
                
        vout.release()
        
        # Save Ground Truth coordinates and metadata
        np.savez_compressed(
            out_gt_path,
            seq_id=seq_id,
            action=action,
            description=description,
            x=x_coords,
            y=y_coords,
            visibility=visibility,
            width=W,
            height=H,
            nframes=len(frame_files)
        )
        
        manifest.append({
            "seq_id": seq_id,
            "action": action,
            "video_path": out_video_path,
            "gt_path": out_gt_path,
            "width": W,
            "height": H,
            "frames": len(frame_files),
            "description": description
        })
        
    print(f"\nSuccessfully curated {len(manifest)} Penn Action videos in '{OUTPUT_DIR}'!")
    print("\nSummary of Curated Dataset:")
    print(f"{'Seq ID':<8} {'Action':<16} {'Resolution':<12} {'Frames':<8} {'Description'}")
    print("-" * 75)
    for m in manifest:
        print(f"{m['seq_id']:<8} {m['action']:<16} {m['width']}x{m['height']:<7} {m['frames']:<8} {m['description']}")


if __name__ == "__main__":
    curate_dataset()
