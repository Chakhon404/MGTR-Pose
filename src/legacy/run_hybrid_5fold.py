import os
import argparse
from ..pipeline.hybrid_sparse import HybridSparseFlowPipeline
from ..utils.video_utils import resolve_video_path, resolve_model_path


def run_5fold_eval(video_paths, model_path, conf, device, output_dir, flow_scale, motion_thr, max_skip):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for fold_idx, video_path in enumerate(video_paths):
        print(f"\n[Fold {fold_idx + 1}/{len(video_paths)}] {os.path.basename(video_path)}")
        pipeline = HybridSparseFlowPipeline(model_path, video_path, conf, device, flow_scale=flow_scale, motion_thr=motion_thr, max_skip=max_skip)
        results = pipeline.run()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"result_hybridM_new_{video_name}.npz")
        pipeline.save_results(output_path)
        all_results.append({"fold": fold_idx, "video": video_name, "fps": results["processing_fps"]})
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--model", default="yolov8m-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--flow_scale", type=float, default=0.25)
    parser.add_argument("--motion_thr", type=float, default=3.0)
    parser.add_argument("--max_skip", type=int, default=6)
    args = parser.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_paths = [resolve_video_path(v, project_root) for v in args.videos]
    model_path = resolve_model_path(args.model, project_root)
    results = run_5fold_eval(video_paths, model_path, args.conf, args.device, args.output, args.flow_scale, args.motion_thr, args.max_skip)
    print("\n=== Summary ===")
    for r in results:
        print(f"Fold {r['fold']}: {r['video']} - {r['fps']:.2f} FPS")


if __name__ == "__main__":
    main()
