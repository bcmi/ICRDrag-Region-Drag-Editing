"""
Compute MD (Motion/Drag Distance) for DragBench-SR dataset and 7 methods.

Dataset: data/DragBench/openvid_format_dbscan_sr — flat structure, each {sample_id}/ has original_image.png and meta_data.pkl.
Results: outputs/dragbench/DragBench-SR/{method}/... (method-specific paths).

Output: one MD result txt per method under MD_regiondrag/ (same dir as this script).
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional

# Project root: script is evaluation/DragBench/DragBench_SR/compute_MD_dragbench_sr.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REGION_DRAG_PATH = os.path.join(PROJECT_ROOT, "evaluation", "third_party")
if REGION_DRAG_PATH not in sys.path:
    sys.path.insert(0, REGION_DRAG_PATH)

from region_utils.evaluator import DragEvaluator

# Default paths (relative to project root)
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data/DragBench/openvid_format_dbscan_sr")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "outputs/dragbench/DragBench-SR")
# Results under this script's directory: DragBench/DragBench_SR/MD_regiondrag
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "MD_regiondrag")

METHODS = [
    "DragDiffusion",
    "DragLoRA",
    "DragonDiffusion",
    "FastDrag",
    "GoodDrag",
    "Inpaint4Drag",
    "SDE-Drag",
]


def collect_samples(source_root: str) -> List[Tuple[str, str]]:
    """
    Walk dragbench-sr (flat structure) and collect (original_image_path, sample_id)
    for each dir that has original_image.png, source_mask.png, and target_mask.png.
    Returns list of (orig_path, sample_id).
    """
    samples = []
    for sample_id in sorted(os.listdir(source_root)):
        sample_path = os.path.join(source_root, sample_id)
        if not os.path.isdir(sample_path):
            continue
        orig_path = os.path.join(sample_path, "original_image.png")
        source_mask = os.path.join(sample_path, "source_mask.png")
        target_mask = os.path.join(sample_path, "target_mask.png")
        if os.path.isfile(orig_path) and os.path.isfile(source_mask) and os.path.isfile(target_mask):
            samples.append((orig_path, sample_id))
    return samples


def mask_center_xy(mask_path: str) -> List[int]:
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [mask.shape[1] // 2, mask.shape[0] // 2]
    return [int(round(xs.mean())), int(round(ys.mean()))]


def load_points_from_meta(meta_path: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Load meta_data.pkl and parse points into source_pts (handle) and target_pts (target).
    points in pkl: list of [x,y], alternating [handle, target, handle, target, ...].
    """
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    points = meta.get("points", [])
    if len(points) < 2:
        return [], []
    source_pts = []
    target_pts = []
    for i in range(0, len(points) - 1, 2):
        h, t = points[i], points[i + 1]
        source_pts.append([int(h[0]), int(h[1])])
        target_pts.append([int(t[0]), int(t[1])])
    return source_pts, target_pts


def load_points_from_sample(sample_dir: str) -> Tuple[List[List[int]], List[List[int]]]:
    meta_path = os.path.join(sample_dir, "meta_data.pkl")
    if os.path.isfile(meta_path):
        return load_points_from_meta(meta_path)
    source_pt = mask_center_xy(os.path.join(sample_dir, "source_mask.png"))
    target_pt = mask_center_xy(os.path.join(sample_dir, "target_mask.png"))
    return [source_pt], [target_pt]


def resolve_edited_path(
    method: str,
    method_root: str,
    sample_id: str,
) -> Optional[str]:
    """
    Resolve edited image path for each method (DragBench-SR flat structure).
    Same logic as run_eval_dragbench_sr.py.
    """
    def _resolve_sr_subdir(mr: str, sid: str, filename: str) -> Optional[str]:
        try:
            subdirs = [d for d in os.listdir(mr) if os.path.isdir(os.path.join(mr, d))]
            for sub in sorted(subdirs):
                path = os.path.join(mr, sub, sid, filename)
                if os.path.isfile(path):
                    return path
        except OSError:
            pass
        return None

    configs = {
        "DragDiffusion": lambda: _resolve_sr_subdir(method_root, sample_id, "dragged_image.png"),
        "DragLoRA": lambda: os.path.join(
            method_root, "drag_results_sr", sample_id, "dragged_image.png"
        ),
        "DragonDiffusion": lambda: os.path.join(
            method_root, "output_dragbench_sr", "edited", f"{sample_id}.jpg"
        ),
        "FastDrag": lambda: _resolve_sr_subdir(method_root, sample_id, "dragged_image.png"),
        "GoodDrag": lambda: os.path.join(
            method_root, "bench_result_sr", sample_id, "output_image.png"
        ),
        "Inpaint4Drag": lambda: os.path.join(
            method_root, "output_dragbench_sr", sample_id, "dragged_image.png"
        ),
        "SDE-Drag": lambda: os.path.join(
            method_root, "output_dragbench_sr", "sdedrag_dragbench", f"{sample_id}.png"
        ),
    }
    if method not in configs:
        return None
    path = configs[method]()
    return path if os.path.isfile(path) else None


def resolve_icrdrag_flat_path(icrdrag_dir: str, sample_id: str) -> Optional[str]:
    for ext in (".jpg", ".png"):
        path = os.path.join(icrdrag_dir, f"{sample_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


def load_image(path: str, edited_panel_index: Optional[int] = None) -> np.ndarray:
    """与 compute_MD_regiondrag 一致：PIL 读图后 np.array。"""
    image = Image.open(path).convert("RGB")
    if edited_panel_index is not None:
        panel_width = image.width // 4
        left = edited_panel_index * panel_width
        image = image.crop((left, 0, left + panel_width, image.height))
    return np.array(image)


def run_md_for_method(
    method: str,
    samples: List[Tuple[str, str]],
    result_root: str,
    output_dir: str,
    sd_method: str,
    evaluator: DragEvaluator,
    icrdrag_dir: Optional[str] = None,
) -> None:
    method_root = os.path.join(result_root, method)
    edited_panel_index = 2 if method == "ICRDrag" and icrdrag_dir else None
    all_distances = []
    results_per_sample = []

    for orig_path, sample_id in tqdm(samples, desc=f"MD {method}"):
        source_pts, target_pts = load_points_from_sample(os.path.dirname(orig_path))
        if not source_pts or not target_pts:
            continue

        edited_path = (
            resolve_icrdrag_flat_path(icrdrag_dir, sample_id)
            if method == "ICRDrag" and icrdrag_dir
            else resolve_edited_path(method, method_root, sample_id)
        )
        if edited_path is None:
            continue

        try:
            ori_image = load_image(orig_path)
            out_image = load_image(edited_path, edited_panel_index)
        except Exception as e:
            print(f"  Skip {sample_id} (load): {e}")
            continue

        prompt = ""
        try:
            md = evaluator.compute_distance(
                ori_image, out_image,
                source_pts, target_pts,
                method=sd_method,
                prompt=prompt if sd_method == "sd" else None,
            )
        except Exception as e:
            print(f"  Skip {sample_id}: {e}")
            continue

        all_distances.append(md)
        results_per_sample.append((sample_id, md))

    if not all_distances:
        print(f"[{method}] No valid samples.")
        return

    mean_md = torch.tensor(all_distances).mean().item()
    out_txt = os.path.join(output_dir, f"{method}_MD.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Method: {method}\n")
        f.write(f"Feature: {sd_method}\n")
        f.write(f"Num samples: {len(all_distances)}\n")
        f.write(f"MD: {mean_md:.4f}\n")
        f.write("\nPer-sample:\n")
        for name, md in results_per_sample:
            f.write(f"  {name}: {md:.4f}\n")
    print(f"[{method}] MD: {mean_md:.4f} (n={len(all_distances)}) -> {out_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute MD on DragBench-SR for all 7 methods."
    )
    parser.add_argument(
        "--source_root", type=str, default=SOURCE_ROOT,
        help="Root of dragbench-sr dataset",
    )
    parser.add_argument(
        "--result_root", type=str, default=RESULT_ROOT,
        help="Root of DragBench-SR results (method subdirs)",
    )
    parser.add_argument(
        "--icrdrag_dir", type=str, default=None,
        help="Flat ICRDrag output dir from inference/run_inference.py.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help="Directory to save MD result txt files",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to evaluate. Default: all 7.",
    )
    parser.add_argument(
        "--sd_method", type=str, default="sd", choices=["sd", "dino"],
        help="Feature method for MD: sd or dino",
    )
    parser.add_argument(
        "--sd_model_path", type=str,
        default=os.environ.get("EVAL_SD_MODEL_PATH") or os.environ.get("EVAL_SD_MODEL_ID"),
        help="Stable Diffusion model path used when --sd_method sd",
    )
    args = parser.parse_args()
    if args.sd_method == "sd" and not args.sd_model_path:
        parser.error("--sd_model_path or EVAL_SD_MODEL_PATH is required when --sd_method sd.")

    methods = ["ICRDrag"] if args.icrdrag_dir else (args.methods if args.methods else METHODS)
    samples = collect_samples(args.source_root)
    if not samples:
        print(f"No samples found under {args.source_root}")
        return

    print(f"Found {len(samples)} samples. Evaluating {len(methods)} methods.")

    evaluator = DragEvaluator(sd_model_path=args.sd_model_path)
    for method in methods:
        run_md_for_method(
            method,
            samples,
            args.result_root,
            args.output_dir,
            args.sd_method,
            evaluator,
            args.icrdrag_dir,
        )

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
