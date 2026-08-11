import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_normalized_jitter(kpts, width, height):
    kpts_norm = kpts.copy()
    kpts_norm[:, :, 0] /= width
    kpts_norm[:, :, 1] /= height
    if len(kpts_norm) < 2:
        return 0.0
    diffs = np.linalg.norm(kpts_norm[1:] - kpts_norm[:-1], axis=-1)
    return np.mean(diffs)


def analyze_jitter_file(npz_path):
    data = np.load(npz_path)
    if 'keypoints_2d' in data:
        kpts = data['keypoints_2d'].reshape(-1, 17, 2)
    elif 'kpts' in data:
        kpts = data['kpts'].reshape(-1, 17, 2)
    else:
        raise KeyError(f"Unknown keys in NPZ: {list(data.keys())}. Expected 'keypoints_2d' or 'kpts'.")
    width = data.get('width', 1000)
    height = data.get('height', 1000)
    jitter = calculate_normalized_jitter(kpts, width, height)
    return {'name': os.path.basename(npz_path).replace('.npz', ''), 'jitter': jitter, 'num_frames': len(kpts)}


def plot_jitter_results(results, output_path):
    names = [r['name'] for r in results]
    values = [r['jitter'] * 100 for r in results]
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color='indianred', alpha=0.8)
    plt.xlabel('Video')
    plt.ylabel('Jitter (x100, normalized)')
    plt.title('Frame-to-Frame Jitter Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved jitter chart to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="jitter_analysis.png")
    args = parser.parse_args()
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.npz')]
    results = []
    for f in files:
        try:
            result = analyze_jitter_file(os.path.join(args.input_dir, f))
            results.append(result)
            print(f"{result['name']}: Jitter = {result['jitter']*100:.4f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
    if results:
        plot_jitter_results(results, args.output)


if __name__ == "__main__":
    main()
