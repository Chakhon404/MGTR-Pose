import os
import argparse
from .pipeline import PureYOLOPipeline, HybridSparseFlowPipeline, HybridDenseFlowPipeline
from .utils.video_utils import resolve_video_path, resolve_model_path, generate_output_path
from .config import PROJECT_ROOT, DEFAULT_MODEL, DEFAULT_VIDEO, DEFAULT_CONF, DEFAULT_DEVICE, OUTPUT_NPZ_DIR


def get_pipeline_class(pipeline_type):
    pipelines = {
        "pure": PureYOLOPipeline,
        "sparse": HybridSparseFlowPipeline,
        "dense": HybridDenseFlowPipeline,
    }
    if pipeline_type not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline_type}. Choose from: {list(pipelines.keys())}")
    return pipelines[pipeline_type]


def main():
    parser = argparse.ArgumentParser(description="MG-Pose: Hybrid Adaptive Skipping System")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--pipeline", type=str, default="sparse", choices=["pure", "sparse", "dense"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--flow_scale", type=float, default=0.25)
    parser.add_argument("--motion_thr", type=float, default=3.0)
    parser.add_argument("--max_skip", type=int, default=6)
    args = parser.parse_args()
    
    video_path = resolve_video_path(args.video, PROJECT_ROOT)
    model_path = resolve_model_path(args.model, PROJECT_ROOT)
    
    if args.output:
        output_path = args.output
    else:
        prefixes = {"pure": "result_pureN_", "sparse": "result_hybridM_new_", "dense": "result_hybridM_"}
        output_path = generate_output_path(args.video, prefixes[args.pipeline], OUTPUT_NPZ_DIR)
    
    print(f"[INFO] Pipeline: {args.pipeline.upper()}")
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Output: {output_path}")
    
    pipeline_class = get_pipeline_class(args.pipeline)
    
    if args.pipeline == "pure":
        pipeline = pipeline_class(model_path, video_path, args.conf, args.device)
    else:
        pipeline = pipeline_class(
            model_path, video_path, args.conf, args.device,
            flow_scale=args.flow_scale,
            motion_thr=args.motion_thr,
            max_skip=args.max_skip
        )
    
    results = pipeline.run()
    pipeline.save_results(output_path)
    print(f"Saved to: {output_path}")
    print(f"Processing Speed: {results['processing_fps']:.2f} FPS")


if __name__ == "__main__":
    main()
