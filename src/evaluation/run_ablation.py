import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run MG-Pose Ablation Experiments")
    parser.add_argument('--ablation', type=str, choices=['all', '1', '2', '3', '4', '5', '6'], default='all')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--videos', nargs='+', default=['Idle.mp4', 'jumping_jack.mp4', 'Idle+jumpimg_jack.mp4'])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    base_out_dir = project_root / 'data' / 'outputs' / 'output_npz' / 'ablation'

    experiments = []

    # Ablation 1: Motion Score Method
    abl1_dir = base_out_dir / 'abl1_motion_method'
    abl1_configs = [
        {'pipeline': 'sparse', 'motion_thr': 3.0, 'prefix': 'sparse_v8n_'},
        {'pipeline': 'dense', 'motion_thr': 0.9, 'prefix': 'dense_v8n_'},
        {'pipeline': 'framediff', 'motion_thr': 3.0, 'prefix': 'framediff_v8n_'}
    ]
    for cfg in abl1_configs:
        cfg.update({'model': 'yolov8n-pose.pt', 'max_skip': 6, 'flow_scale': 0.25, 'out_dir': abl1_dir, 'ablation': '1'})
        experiments.append(cfg)

    # Ablation 2: Threshold Sweep
    abl2_dir = base_out_dir / 'abl2_threshold'
    abl2_configs = [
        {'motion_thr': 1.0, 'prefix': 'thr1.0_v8n_'},
        {'motion_thr': 2.0, 'prefix': 'thr2.0_v8n_'},
        {'motion_thr': 3.0, 'prefix': 'thr3.0_v8n_'},
        {'motion_thr': 5.0, 'prefix': 'thr5.0_v8n_'},
        {'motion_thr': 7.0, 'prefix': 'thr7.0_v8n_'},
        {'motion_thr': 10.0, 'prefix': 'thr10.0_v8n_'}
    ]
    for cfg in abl2_configs:
        cfg.update({'pipeline': 'sparse', 'model': 'yolov8n-pose.pt', 'max_skip': 6, 'flow_scale': 0.25, 'out_dir': abl2_dir, 'ablation': '2'})
        experiments.append(cfg)

    # Ablation 3: Max Skip Strategy
    abl3_dir = base_out_dir / 'abl3_maxskip'
    abl3_configs = [
        {'max_skip': 2, 'prefix': 'skip2_v8n_'},
        {'max_skip': 4, 'prefix': 'skip4_v8n_'},
        {'max_skip': 6, 'prefix': 'skip6_v8n_'},
        {'max_skip': 10, 'prefix': 'skip10_v8n_'},
        {'max_skip': 15, 'prefix': 'skip15_v8n_'}
    ]
    for cfg in abl3_configs:
        cfg.update({'pipeline': 'sparse', 'model': 'yolov8n-pose.pt', 'motion_thr': 3.0, 'flow_scale': 0.25, 'out_dir': abl3_dir, 'ablation': '3'})
        experiments.append(cfg)

    # Ablation 4: Temporal Refinement
    abl4_dir = base_out_dir / 'abl4_refinement'
    abl4_configs = [
        {'no_interp': False, 'prefix': 'with_interp_v8n_'},
        {'no_interp': True, 'prefix': 'no_interp_v8n_'}
    ]
    for cfg in abl4_configs:
        cfg.update({'pipeline': 'sparse', 'model': 'yolov8n-pose.pt', 'motion_thr': 3.0, 'max_skip': 6, 'flow_scale': 0.25, 'out_dir': abl4_dir, 'ablation': '4'})
        experiments.append(cfg)

    # Ablation 5: Flow Scale
    abl5_dir = base_out_dir / 'abl5_flowscale'
    abl5_configs = [
        {'flow_scale': 1.00, 'prefix': 'scale1.00_v8n_'},
        {'flow_scale': 0.50, 'prefix': 'scale0.50_v8n_'},
        {'flow_scale': 0.25, 'prefix': 'scale0.25_v8n_'}
    ]
    for cfg in abl5_configs:
        cfg.update({'pipeline': 'sparse', 'model': 'yolov8n-pose.pt', 'motion_thr': 3.0, 'max_skip': 6, 'out_dir': abl5_dir, 'ablation': '5'})
        experiments.append(cfg)

    # Ablation 6: Model Backbone
    abl6_dir = base_out_dir / 'abl6_model'
    abl6_configs = [
        {'pipeline': 'pure', 'model': 'yolov8n-pose.pt', 'prefix': 'pure_v8n_'},
        {'pipeline': 'sparse', 'model': 'yolov8n-pose.pt', 'prefix': 'hybrid_v8n_'},
        {'pipeline': 'pure', 'model': 'yolov8m-pose.pt', 'prefix': 'pure_v8m_'},
        {'pipeline': 'sparse', 'model': 'yolov8m-pose.pt', 'prefix': 'hybrid_v8m_'},
        {'pipeline': 'pure', 'model': 'yolo26n-pose.pt', 'prefix': 'pure_v26n_'},
        {'pipeline': 'sparse', 'model': 'yolo26n-pose.pt', 'prefix': 'hybrid_v26n_'},
        {'pipeline': 'pure', 'model': 'yolo26m-pose.pt', 'prefix': 'pure_v26m_'},
        {'pipeline': 'sparse', 'model': 'yolo26m-pose.pt', 'prefix': 'hybrid_v26m_'}
    ]
    for cfg in abl6_configs:
        cfg.update({'motion_thr': 3.0, 'max_skip': 6, 'flow_scale': 0.25, 'out_dir': abl6_dir, 'ablation': '6'})
        experiments.append(cfg)

    runs = []
    for exp in experiments:
        if args.ablation != 'all' and exp['ablation'] != args.ablation:
            continue
        for video in args.videos:
            video_name = Path(video).stem
            out_file = exp['out_dir'] / f"{exp['prefix']}{video_name}.npz"
            run_info = exp.copy()
            run_info['video'] = video
            run_info['out_file'] = out_file
            runs.append(run_info)

    total_runs = len(runs)
    print(f"Total runs scheduled: {total_runs}")

    for idx, run in enumerate(runs, 1):
        if run['out_file'].exists():
            print(f"[{idx}/{total_runs}] Skipping {run['prefix']} on {run['video']}, file exists.")
            continue
            
        print(f"[{idx}/{total_runs}] Running {run['prefix']} on {run['video']}...")
        run['out_dir'].mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, '-m', 'src.main',
            '--pipeline', run['pipeline'],
            '--video', f"data/inputs/{run['video']}",
            '--model', f"models/{run['model']}",
            '--device', args.device,
            '--output', str(run['out_file'])
        ]
        
        if 'motion_thr' in run and run['pipeline'] != 'pure':
            cmd.extend(['--motion_thr', str(run['motion_thr'])])
        if 'max_skip' in run and run['pipeline'] != 'pure':
            cmd.extend(['--max_skip', str(run['max_skip'])])
        if 'flow_scale' in run and run['pipeline'] != 'pure':
            cmd.extend(['--flow_scale', str(run['flow_scale'])])
        if run.get('no_interp', False):
            cmd.append('--no_interp')

        subprocess.run(cmd, cwd=project_root)

if __name__ == '__main__':
    main()
