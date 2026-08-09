import os
import argparse

from .utils.render import render_video, get_output_path
from .config import PROJECT_ROOT


def resolve_path(path, base_dir, default_search_dirs):
    """Resolve path - check absolute, then relative to search directories."""
    if os.path.exists(path):
        return path
    
    for search_dir in default_search_dirs:
        full_path = os.path.join(search_dir, path)
        if os.path.exists(full_path):
            return full_path
    
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Render pose overlay on video from NPZ results"
    )
    
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Input video file"
    )
    
    parser.add_argument(
        "--npz", "-n",
        required=True,
        help="NPZ file with pose results"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output MP4 path (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["raw", "refined"],
        default=None,
        help="Pose visualization style: raw (green/yellow) or refined (purple/blue). Auto-detected if not specified"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=None,
        help="Output FPS (uses video FPS by default)"
    )
    
    args = parser.parse_args()
    
    project_root = PROJECT_ROOT
    
    video_path = resolve_path(
        args.video,
        project_root,
        [project_root, os.path.join(project_root, "data", "inputs")]
    )
    
    npz_path = resolve_path(
        args.npz,
        project_root,
        [project_root, os.path.join(project_root, "data", "outputs", "output_npz")]
    )
    
    output_path = args.output or get_output_path(npz_path, args.output, project_root)
    
    print("=" * 50)
    print("MG-Pose: NPZ to Video Renderer")
    print("=" * 50)
    print(f"Video:   {video_path}")
    print(f"NPZ:     {npz_path}")
    print(f"Output:  {output_path}")
    print(f"Style:   {args.style or 'auto'}")
    print(f"FPS:     {args.fps or 'video default'}")
    print("=" * 50)
    
    result = render_video(
        video_path=video_path,
        npz_path=npz_path,
        output_path=output_path,
        style=args.style,
        fps_override=args.fps
    )
    
    print("=" * 50)
    print(f"Done! Rendered {result['frames_rendered']} frames")
    print(f"Style: {result['style']}")
    print(f"Output: {result['output_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
