import os
import numpy as np
import matplotlib.pyplot as plt


def load_and_scale_keypoints(npz_path, target_w=1000, target_h=1000):
    data = np.load(npz_path)
    kpts = data['keypoints_2d'].reshape(-1, 17, 2)
    if 'width' in data and 'height' in data:
        orig_w, orig_h = data['width'], data['height']
    else:
        orig_w, orig_h = 1920, 1080
    scaled = kpts.copy()
    scaled[:, :, 0] *= (target_w / orig_w)
    scaled[:, :, 1] *= (target_h / orig_h)
    return scaled


def calculate_mpjpe(pred_kpts, gt_kpts):
    errors = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)
    return np.mean(errors)


def evaluate_pair(pred_path, gt_path):
    pred_kpts = load_and_scale_keypoints(pred_path)
    gt_kpts = load_and_scale_keypoints(gt_path)
    min_len = min(len(pred_kpts), len(gt_kpts))
    return calculate_mpjpe(pred_kpts[:min_len], gt_kpts[:min_len])


def plot_mpjpe_results(results, output_path):
    names = [r['name'] for r in results]
    values = [r['mpjpe'] for r in results]
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color='teal', alpha=0.8)
    plt.xlabel('Video')
    plt.ylabel('MPJPE (pixels)')
    plt.title('Mean Per Joint Position Error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved MPJPE chart to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output", default="mpjpe_results.png")
    args = parser.parse_args()
    pred_files = {f.replace('.npz', ''): f for f in os.listdir(args.pred_dir) if f.endswith('.npz')}
    gt_files = {f.replace('.npz', ''): f for f in os.listdir(args.gt_dir) if f.endswith('.npz')}
    results = []
    for name, pred_file in pred_files.items():
        if name in gt_files:
            try:
                mpjpe = evaluate_pair(os.path.join(args.pred_dir, pred_file), os.path.join(args.gt_dir, gt_files[name]))
                results.append({'name': name, 'mpjpe': mpjpe})
                print(f"{name}: MPJPE = {mpjpe:.2f} px")
            except Exception as e:
                print(f"Error processing {name}: {e}")
    if results:
        plot_mpjpe_results(results, args.output)


if __name__ == "__main__":
    main()
