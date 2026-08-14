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
    # The keypoints from YOLO's .xyn are already normalized to [0, 1].
    # DO NOT divide by w and h again.
    pred_norm = pred_kpts 
    if len(pred_norm) < 3:
        return 0.0
    # Calculate Mean Acceleration (Second Derivative)
    accel = np.linalg.norm(pred_norm[2:] - 2*pred_norm[1:-1] + pred_norm[:-2], axis=-1)
    # Handle NaNs that might exist if a person is completely lost
    jitter = np.nanmean(accel)
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
                w = float(data['width']) if 'width' in data else 1920.0
                h = float(data['height']) if 'height' in data else 1080.0
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
            if abl == 'abl6_model':
                order = {
                    'pure_v8n': 1, 'hybrid_v8n': 2,
                    'pure_v8m': 3, 'hybrid_v8m': 4,
                    'pure_v26n': 5, 'hybrid_v26n': 6,
                    'pure_v26m': 7, 'hybrid_v26m': 8
                }
                for k, v in order.items():
                    if k in s:
                        return v
                return 99
                
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
        try:
            plt.style.use('seaborn-v0_8-muted')
        except:
            pass # fallback if not available
            
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        metrics = [
            ('FPS', 'steelblue', 'Inference Speed: FPS Comparison\n(Higher is Better)'),
            ('MPJPE', 'teal', 'Accuracy Evaluation: MPJPE\n(Lower is Better)'),
            ('Jitter', 'indianred', 'Smoothness Evaluation: Jitter Score\n(Lower is Better)')
        ]
        
        import matplotlib.colors as mcolors
        for ax, (metric, color, title) in zip(axes, metrics):
            values = summary_df[metric]
            if metric == 'Jitter':
                values = values * 1000  # Scale jitter by 10^3
                ax.set_ylabel('Mean Acceleration ($10^{-3}$ Normalized Units)', fontsize=11, fontweight='bold')
            elif metric == 'MPJPE':
                values = values * 1000  # Scale MPJPE by 10^3
                ax.set_ylabel('MPJPE ($10^{-3}$ Normalized Units)', fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel('Frames Per Second (FPS)', fontsize=11, fontweight='bold')
                
            base_rgba = mcolors.to_rgba(color)
            bar_colors = []
            for cfg in summary_df['Configuration']:
                if 'pure' in cfg or 'dense' in cfg or 'framediff' in cfg:
                    bar_colors.append('#6c757d') # Gray for baseline
                elif 'v8n' in cfg or 'v26n' in cfg:
                    bar_colors.append('#dc3545') # Red for Hybrid-N
                else:
                    bar_colors.append('#007bff') # Blue for Hybrid-M
                    
            bars = ax.bar(summary_df['Configuration'], values, color=bar_colors, edgecolor='black', linewidth=1.2, width=0.8)
            
            # Add values on top of bars
            for bar in bars:
                yval = bar.get_height()
                label = f"{yval:.2f}"
                ax.text(bar.get_x() + bar.get_width()/2., yval + (ax.get_ylim()[1]*0.02), label, ha='center', va='bottom', fontsize=11, fontweight='bold')
                
            # Increase Y limit slightly to make room for text
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)

            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            
            # Custom labels for Ablation 6
            labels = []
            for cfg in summary_df['Configuration']:
                if cfg == 'pure_v8n': labels.append('YOLOv8n\n(Baseline)')
                elif cfg == 'hybrid_v8n': labels.append('Hybrid-N\n(Ours)')
                elif cfg == 'pure_v8m': labels.append('YOLOv8m\n(Baseline)')
                elif cfg == 'hybrid_v8m': labels.append('Hybrid-M\n(Ours)')
                elif cfg == 'pure_v26n': labels.append('YOLOv26n\n(Baseline)')
                elif cfg == 'hybrid_v26n': labels.append('Hybrid-26N\n(Ours)')
                elif cfg == 'pure_v26m': labels.append('YOLOv26m\n(Baseline)')
                elif cfg == 'hybrid_v26m': labels.append('Hybrid-26M\n(Ours)')
                else: labels.append(cfg)
                
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10, fontweight='bold', rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.4, linestyle='--')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(output_dir / f"{abl}_chart.png", dpi=150)
        plt.close()

if __name__ == '__main__':
    main()
