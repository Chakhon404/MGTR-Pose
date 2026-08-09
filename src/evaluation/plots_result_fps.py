import os
import numpy as np
import matplotlib.pyplot as plt


def load_fps_results(npz_dir):
    files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    results = []
    for f in files:
        data = np.load(os.path.join(npz_dir, f))
        results.append({
            'name': f.replace('.npz', ''),
            'video_fps': data.get('video_fps', 0),
            'proc_fps': data.get('processing_fps', 0)
        })
    return results


def plot_fps_comparison(results, output_path):
    names = [r['name'] for r in results]
    video_fps = [r['video_fps'] for r in results]
    proc_fps = [r['proc_fps'] for r in results]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, video_fps, width, label='Video FPS', color='steelblue')
    ax.bar(x + width/2, proc_fps, width, label='Processing FPS', color='coral')
    ax.set_xlabel('Video')
    ax.set_ylabel('FPS')
    ax.set_title('Processing Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved FPS chart to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="fps_comparison.png")
    args = parser.parse_args()
    results = load_fps_results(args.input_dir)
    plot_fps_comparison(results, args.output)


if __name__ == "__main__":
    main()
