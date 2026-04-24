"""
Composite multiple videos in a folder into a single blended output.

Usage:
    python make_composite.py <input_folder> [output_path] [--boost FLOAT] [--cube-alpha FLOAT] [--fps INT]

Defaults:
    output_path  = <input_folder>/composite.mp4
    --boost      = 2.5   (contrast boost on composite)
    --cube-alpha = 0.4   (nudge toward per-pixel max; lifts moving objects)
    --fps        = 60
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image


def load_video(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if path.is_dir():
        sys.exit(f'Error: output path {path} is a directory — pass a .mp4 filename, e.g. {path}/composite.mp4')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.remove(path)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8, macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()


def composite_fg_boost(
    frames_list: list[list[np.ndarray]],
    boost: float = 2.5,
    cube_alpha: float = 0.4,
) -> list[np.ndarray]:
    max_len = max(len(f) for f in frames_list)
    h, w = frames_list[0][0].shape[:2]
    def pad_and_resize(frames):
        out = []
        for f in frames:
            if f.shape[0] != h or f.shape[1] != w:
                f = np.array(Image.fromarray(f).resize((w, h), Image.LANCZOS))
            out.append(f)
        while len(out) < max_len:
            out.append(out[-1])
        return np.stack(out).astype(np.float32)
    padded = [pad_and_resize(f) for f in frames_list]

    stack = np.stack(padded, axis=0)   # (N, T, H, W, 3)
    mean = stack.mean(axis=0)          # (T, H, W, 3)
    maximum = stack.max(axis=0)        # (T, H, W, 3)

    composite = mean + cube_alpha * (maximum - mean)
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return list(composite)


def main() -> None:
    parser = argparse.ArgumentParser(description='Composite videos in a folder.')
    parser.add_argument('input_folder', type=Path, help='Folder containing .mp4 files')
    parser.add_argument('output_path', type=Path, nargs='?', help='Output .mp4 path (default: <input_folder>/composite.mp4)')
    parser.add_argument('--boost', type=float, default=2.5, help='Contrast boost (unused legacy param, kept for compat)')
    parser.add_argument('--cube-alpha', type=float, default=0.0, dest='cube_alpha', help='Weight toward per-pixel max (0=pure mean, 1=pure max)')
    parser.add_argument('--fps', type=int, default=60, help='Output FPS')
    args = parser.parse_args()

    folder: Path = args.input_folder
    if not folder.is_dir():
        sys.exit(f'Error: {folder} is not a directory')

    video_paths = sorted(folder.glob('*.mp4'))
    if not video_paths:
        sys.exit(f'Error: no .mp4 files found in {folder}')

    out_path: Path = args.output_path or folder / 'composite.mp4'
    if out_path in video_paths:
        video_paths = [p for p in video_paths if p != out_path]

    print(f'Found {len(video_paths)} videos in {folder}')
    for p in video_paths:
        print(f'  {p.name}')

    print('Loading videos...', flush=True)
    frames_list = []
    for p in video_paths:
        print(f'  loading {p.name}...', end=' ', flush=True)
        frames = load_video(p)
        frames_list.append(frames)
        print(f'{len(frames)} frames')

    print('Compositing...', flush=True)
    composite = composite_fg_boost(frames_list, boost=args.boost, cube_alpha=args.cube_alpha)

    print(f'Saving to {out_path}...', flush=True)
    save_video(composite, out_path, args.fps)
    print(f'Done. {len(composite)} frames @ {args.fps} fps -> {out_path}')


if __name__ == '__main__':
    main()
