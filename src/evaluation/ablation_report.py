import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def compute_mpjpe(pred_kpts, gt_kpts, pred_w, pred_h, gt_w, gt_h):
    if pred_kpts.ndim == 2:
        pred_kpts = pred_kpts.reshape(pred_kpts.shape[0], -1, 2)
    if gt_kpts.ndim == 2:
        gt_kpts = gt_kpts.reshape(gt_kpts.shape[0], -1, 2)
        
    min_len = min(len(pred_kpts), len(gt_kpts))
    pred = pred_kpts[:min_len]
    gt = gt_kpts[:min_len]
    
    pred_norm = pred / np.array([pred_w, pred_h])
    gt_norm = gt / np.array([gt_w, gt_h])
    
    gt_scaled = gt_norm * np.array([pred_w, pred_h])
    
    mpjpe = np.mean(np.linalg.norm(pred - gt_scaled, axis=-1))
    return mpjpe

def compute_jitter(pred_kpts, w, h):
    if pred_kpts.ndim == 2:
        pred_kpts = pred_kpts.reshape(pred_kpts.shape[0], -1, 2)
    pred_norm = pred_kpts / np.array([w, h])
    jitter = np.mean(np.linalg.norm(pred_norm[1:] - pred_norm[:-1], axis=-1))
    return jitter

def load_kpts(npz_data):
    if 'keypoints_2d' in npz_data:
        return npz_data['keypoints_2d']
    elif 'kpts' in npz_data:
        return npz_data['kpts']
    else:
        raise KeyError("Keypoints not found in NPZ file.")

def main():
    parser = argparse.ArgumentParser(description="Generate MG-Pose Ablation Report")
    parser.add_argument('--input_dir', type=str, default='data/outputs/output_npz/ablation')
    parser.add_argument('--gt_dir', type=str, default='data/outputs/output_npz')
    parser.add_argument('--output_dir', type=str, default='data/outputs/ablation_results')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / args.input_dir
    gt_dir = project_root / args.gt_dir
    output_dir = project_root / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = ['abl1_motion_method', 'abl2_threshold', 'abl3_maxskip', 'abl4_refinement', 'abl5_flowscale', 'abl6_model']

    for abl in ablations:
        abl_path = input_dir / abl
        if not abl_path.exists():
            print(f"Skipping {abl} (Directory not found: {abl_path})")
            continue
            
        print(f"\n--- Processing {abl} ---")
        results = []
        
        for npz_file in abl_path.glob('*.npz'):
            try:
                data = np.load(npz_file)
                kpts = load_kpts(data)
                fps = float(data['processing_fps'])
                w, h = float(data['width']), float(data['height'])
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
                continue
            
            # Identify video name from filename
            if 'Idle+jumpimg_jack' in npz_file.stem:
                video_name = 'Idle+jumpimg_jack'
            elif 'jumping_jack' in npz_file.stem:
                video_name = 'jumping_jack'
            elif 'Idle' in npz_file.stem:
                video_name = 'Idle'
            else:
                continue

            # Determine which GT to use
            if 'v26' in npz_file.stem:
                gt_model = 'yolo26x'
                gt_folder = 'gt26'
            else:
                gt_model = 'yolo8x'
                gt_folder = 'gt8'

            gt_file = gt_dir / gt_folder / f"{gt_model}_{video_name}.npz"
            if not gt_file.exists():
                print(f"GT not found for {npz_file.name}: {gt_file}")
                mpjpe = np.nan
            else:
                try:
                    gt_data = np.load(gt_file)
                    gt_kpts = load_kpts(gt_data)
                    gt_w = float(gt_data['width']) if 'width' in gt_data else w
                    gt_h = float(gt_data['height']) if 'height' in gt_data else h
                    mpjpe = compute_mpjpe(kpts, gt_kpts, w, h, gt_w, gt_h)
                except Exception as e:
                    print(f"Error loading GT for {npz_file.name}: {e}")
                    mpjpe = np.nan
                
            jitter = compute_jitter(kpts, w, h)
            
            # Extract configuration from filename
            config_name = npz_file.stem.replace(f"_{video_name}", "")
            
            results.append({
                'Configuration': config_name,
                'Video': video_name,
                'FPS': fps,
                'MPJPE': mpjpe,
                'Jitter': jitter
            })
            
        if not results:
            continue
            
        df = pd.DataFrame(results)
        
        import re
        def extract_sort_key(s):
            # Extract first number from the string (e.g. 'thr10.0_v8n' -> 10.0)
            nums = re.findall(r'\d+\.\d+|\d+', s)
            if nums:
                return float(nums[0])
            return 0.0
            
        summary_df = df.groupby('Configuration').agg({
            'FPS': 'mean',
            'MPJPE': 'mean',
            'Jitter': 'mean'
        }).reset_index()
        
        # Sort properly based on the numeric value inside the configuration name
        summary_df['SortKey'] = summary_df['Configuration'].apply(extract_sort_key)
        summary_df = summary_df.sort_values('SortKey').drop(columns=['SortKey'])
        
        print(f"\nSummary for {abl}:")
        print(summary_df.to_string(index=False))
        
        summary_df.to_csv(output_dir / f"{abl}_summary.csv", index=False)
        df.to_csv(output_dir / f"{abl}_full.csv", index=False)
        
        # Plot Charts
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        ablation_titles = {
            'abl1_motion_method': 'Ablation 1: Motion Score Method Comparison',
            'abl2_threshold': 'Ablation 2: Threshold Sweep',
            'abl3_maxskip': 'Ablation 3: Max Skip Strategy',
            'abl4_refinement': 'Ablation 4: Temporal Refinement',
            'abl5_flowscale': 'Ablation 5: Flow Scale Downsampling',
            'abl6_model': 'Ablation 6: Model Backbone Comparison'
        }
        main_title = ablation_titles.get(abl, f'{abl} Results')
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        metrics = [
            ('FPS', 'steelblue', 'Processing Speed (FPS)'),
            ('MPJPE', 'teal', 'Mean Per Joint Position Error'),
            ('Jitter', 'indianred', 'Frame-to-Frame Jitter')
        ]
        
        for ax, (metric, color, title) in zip(axes, metrics):
            values = summary_df[metric]
            if metric == 'Jitter':
                values = values * 100  # Scale jitter like original plot
                ax.set_ylabel('Jitter (x100, normalized)')
            elif metric == 'MPJPE':
                ax.set_ylabel('MPJPE (pixels)')
            else:
                ax.set_ylabel('FPS')
                
            ax.bar(summary_df['Configuration'], values, color=color, alpha=0.9)
            ax.set_title(title, fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(axis='y', alpha=0.3)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(output_dir / f"{abl}_chart.png", dpi=150)
        plt.close()

if __name__ == '__main__':
    main()
