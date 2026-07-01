"""
Compute m-MD (masked Mean Distance) for DragBench-DR dataset using DIFT.

Dataset: data/DragBench/openvid_format_dbscan_dr — each sample has original_image.png and meta_data.pkl
  (points: [handle, target, ...], mask: HxW binary).
Results: outputs/dragbench/DragBench-DR/{method}/...
Output: MD_DragLora/{method}_mMD.txt
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm
from typing import Optional, List, Tuple

# Project root: script is evaluation/DragBench/DragBench_DR/compute_mMD_dragbench_dr.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
THIRD_PARTY_PATH = os.path.join(PROJECT_ROOT, "evaluation", "third_party")
if THIRD_PARTY_PATH not in sys.path:
    sys.path.insert(0, THIRD_PARTY_PATH)
DRAGLORA_EVAL_PATH = os.path.join(PROJECT_ROOT, "DragLoRA", "drag_bench_evaluation")
if os.path.isdir(DRAGLORA_EVAL_PATH) and DRAGLORA_EVAL_PATH not in sys.path:
    sys.path.insert(0, DRAGLORA_EVAL_PATH)

from dift_sd import SDFeaturizer  # type: ignore[import-not-found]

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data/DragBench/openvid_format_dbscan_dr")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "outputs/dragbench/DragBench-DR")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "MD_DragLora")

METHODS = [
    "DragDiffusion",
    "DragLoRA",
    "DragonDiffusion",
    "FastDrag",
    "GoodDrag",
    "Inpaint4Drag",
    "SDE-Drag",
]


def collect_samples(source_root: str) -> List[Tuple[str, str, str]]:
    """
    Walk dragbench-dr and collect (original_image_path, category, sample_id) for each leaf dir
    that has original_image.png, source_mask.png, and target_mask.png.
    """
    samples = []
    for cat in sorted(os.listdir(source_root)):
        cat_path = os.path.join(source_root, cat)
        if not os.path.isdir(cat_path):
            continue
        for sample_id in sorted(os.listdir(cat_path)):
            sample_path = os.path.join(cat_path, sample_id)
            if not os.path.isdir(sample_path):
                continue
            orig_path = os.path.join(sample_path, "original_image.png")
            source_mask = os.path.join(sample_path, "source_mask.png")
            target_mask = os.path.join(sample_path, "target_mask.png")
            if os.path.isfile(orig_path) and os.path.isfile(source_mask) and os.path.isfile(target_mask):
                samples.append((orig_path, cat, sample_id))
    return samples


def mask_center_yx(mask_path: str) -> Tuple[int, int]:
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    return int(round(ys.mean())), int(round(xs.mean()))


def load_points_from_meta(meta_path: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Load meta_data.pkl and parse points into handle_pts and target_pts.
    points in pkl: list of [x,y], alternating [handle, target, handle, target, ...].
    Returns (handle_pts, target_pts) each as list of (y, x) for indexing.
    """
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    points = meta.get("points", [])
    if len(points) < 2:
        return [], []
    handle_pts = []
    target_pts = []
    for i in range(0, len(points) - 1, 2):
        h, t = points[i], points[i + 1]
        # Store as (y, x) for tensor indexing
        handle_pts.append((int(h[1]), int(h[0])))
        target_pts.append((int(t[1]), int(t[0])))
    return handle_pts, target_pts


def load_points_from_sample(sample_dir: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    meta_path = os.path.join(sample_dir, "meta_data.pkl")
    if os.path.isfile(meta_path):
        return load_points_from_meta(meta_path)
    handle_pt = mask_center_yx(os.path.join(sample_dir, "source_mask.png"))
    target_pt = mask_center_yx(os.path.join(sample_dir, "target_mask.png"))
    return [handle_pt], [target_pt]


def load_eval_mask(
    sample_dir: str,
    target_size: Tuple[int, int],
    white_threshold: int = 0,
) -> np.ndarray:
    """
    Load mask from meta_data.pkl. mask is (H,W) uint8, 1=mask region.
    Resize to target_size if needed. Returns binary (H, W) uint8.
    """
    meta_path = os.path.join(sample_dir, "meta_data.pkl")
    if os.path.isfile(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        mask = meta.get("mask", None)
    else:
        mask_path = os.path.join(sample_dir, "target_mask.png")
        mask = np.array(Image.open(mask_path).convert("L")) if os.path.isfile(mask_path) else None
    if mask is None:
        return np.ones(target_size, dtype=np.uint8)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > white_threshold).astype(np.uint8)
    h, w = target_size
    if mask.shape[:2] != (h, w):
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask = np.array(mask_pil)
    return mask


def resolve_edited_path(
    method: str,
    method_root: str,
    category: str,
    sample_id: str,
) -> Optional[str]:
    """Resolve edited image path for each method."""
    def _resolve_fastdrag(mr: str, c: str, s: str) -> Optional[str]:
        try:
            for sub in sorted(os.listdir(mr)):
                sub_path = os.path.join(mr, sub)
                if not os.path.isdir(sub_path):
                    continue
                path = os.path.join(sub_path, c, s, "dragged_image.png")
                if os.path.isfile(path):
                    return path
        except OSError:
            pass
        return None

    configs = {
        "DragDiffusion": lambda: os.path.join(
            method_root, category, sample_id, "dragged_image.png"
        ),
        "DragLoRA": lambda: os.path.join(
            method_root, "drag_results", category, sample_id, "dragged_image.png"
        ),
        "DragonDiffusion": lambda: os.path.join(
            method_root, "output_dragbench_dr", "edited", f"{category}_{sample_id}.jpg"
        ),
        "FastDrag": lambda: _resolve_fastdrag(method_root, category, sample_id),
        "GoodDrag": lambda: os.path.join(
            method_root, "bench_result_dr", f"{category}_{sample_id}", "output_image.png"
        ),
        "Inpaint4Drag": lambda: os.path.join(
            method_root, "output_dragbench_dr", category, sample_id, "dragged_image.png"
        ),
        "SDE-Drag": lambda: os.path.join(
            method_root, "output_dragbench_dr", "sdedrag_dragbench",
            category, sample_id, "dragged_image.png"
        ),
    }
    if method not in configs:
        return None
    path = configs[method]()
    return path if os.path.isfile(path) else None


def resolve_icrdrag_flat_path(icrdrag_dir: str, category: str, sample_id: str) -> Optional[str]:
    stems = (f"{category}_{sample_id}", sample_id)
    for ext in (".jpg", ".png"):
        for stem in stems:
            path = os.path.join(icrdrag_dir, f"{stem}{ext}")
            if os.path.isfile(path):
                return path
    return None


def crop_edited_panel(image: Image.Image) -> Image.Image:
    panel_width = image.width // 4
    left = 2 * panel_width
    return image.crop((left, 0, left + panel_width, image.height))


def run_mmd_for_method(
    method: str,
    samples: List[Tuple[str, str, str]],
    result_root: str,
    output_dir: str,
    dift: SDFeaturizer,
    cos: nn.Module,
    default_prompt: str,
    save_per_sample: bool,
    icrdrag_dir: Optional[str] = None,
) -> None:
    method_root = os.path.join(result_root, method)
    use_flat_icrdrag = method == "ICRDrag" and icrdrag_dir
    all_dist_m = []
    results_per_sample = []

    for orig_path, category, sample_id in tqdm(samples, desc=f"m-MD {method}"):
        sample_dir = os.path.dirname(orig_path)
        meta_path = os.path.join(sample_dir, "meta_data.pkl")

        handle_pts, target_pts = load_points_from_sample(sample_dir)
        if not handle_pts or not target_pts:
            continue

        edited_path = (
            resolve_icrdrag_flat_path(icrdrag_dir, category, sample_id)
            if use_flat_icrdrag
            else resolve_edited_path(method, method_root, category, sample_id)
        )
        if edited_path is None:
            continue

        try:
            source_pil = Image.open(orig_path).convert("RGB")
            edited_pil = Image.open(edited_path).convert("RGB")
            if use_flat_icrdrag:
                edited_pil = crop_edited_panel(edited_pil)
        except Exception as e:
            print(f"  Skip {category}/{sample_id} (load): {e}")
            continue

        H, W = source_pil.size[1], source_pil.size[0]
        edited_pil = edited_pil.resize((W, H), Image.BILINEAR)

        # Load mask from meta_data.pkl and resize to image size
        mask_np = load_eval_mask(sample_dir, (H, W))
        mask = torch.from_numpy(mask_np)
        mask_flat = mask.flatten()

        # Clamp points to image bounds
        handle_points = []
        target_points = []
        for (hy, hx), (ty, tx) in zip(handle_pts, target_pts):
            hy_c = int(round(np.clip(hy, 0, H - 1)))
            hx_c = int(round(np.clip(hx, 0, W - 1)))
            ty_c = int(round(np.clip(ty, 0, H - 1)))
            tx_c = int(round(np.clip(tx, 0, W - 1)))
            handle_points.append((hy_c, hx_c))
            target_points.append((ty_c, tx_c))

        # Get prompt from meta if available
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            prompt = meta.get("prompt", default_prompt)
            if not prompt or not isinstance(prompt, str):
                prompt = default_prompt
        except Exception:
            prompt = default_prompt

        # Image tensors: (1, 3, H, W), range [-1, 1]
        source_tensor = (PILToTensor()(source_pil) / 255.0 - 0.5) * 2
        if source_tensor.dim() == 3:
            source_tensor = source_tensor.unsqueeze(0)
        dragged_tensor = (PILToTensor()(edited_pil) / 255.0 - 0.5) * 2
        if dragged_tensor.dim() == 3:
            dragged_tensor = dragged_tensor.unsqueeze(0)
        if source_tensor.shape[2:] != (H, W):
            source_tensor = F.interpolate(source_tensor, (H, W), mode="bilinear")
            dragged_tensor = F.interpolate(dragged_tensor, (H, W), mode="bilinear")

        with torch.no_grad():
            ft_source = dift.forward(
                source_tensor,
                prompt=prompt,
                t=261,
                up_ft_index=1,
                ensemble_size=8,
            )
            ft_source = F.interpolate(ft_source, (H, W), mode="bilinear")
            ft_dragged = dift.forward(
                dragged_tensor,
                prompt=prompt,
                t=261,
                up_ft_index=1,
                ensemble_size=8,
            )
            ft_dragged = F.interpolate(ft_dragged, (H, W), mode="bilinear")

        # Candidate coords within mask
        y_coords = torch.arange(H)
        x_coords = torch.arange(W)
        x_coords = x_coords.repeat(H)
        y_coords = y_coords.repeat_interleave(W)
        grid_coords = torch.stack((y_coords, x_coords), dim=1)
        candidate_coords = grid_coords[mask_flat == 1]
        if candidate_coords.numel() == 0:
            candidate_coords = grid_coords
        ft_dragged_mask = ft_dragged[
            :, :, candidate_coords[:, 0], candidate_coords[:, 1]
        ]

        sample_dists = []
        for pt_idx in range(len(handle_points)):
            hy, hx = handle_points[pt_idx]
            ty, tx = target_points[pt_idx]
            tp = torch.tensor([ty, tx], dtype=torch.float32)
            num_channel = ft_source.size(1)
            src_vec = ft_source[0, :, hy, hx].view(1, num_channel, 1, 1)
            cos_map = cos(src_vec.squeeze(-1), ft_dragged_mask)[0]
            max_rc_m = candidate_coords[cos_map.argmax()]
            dist_m = (tp - max_rc_m.float()).norm()
            sample_dists.append(dist_m.item())
        sample_mmd = np.mean(sample_dists)
        all_dist_m.append(sample_mmd)
        sample_name = f"{category}/{sample_id}"
        results_per_sample.append((sample_name, sample_mmd))

    if not all_dist_m:
        print(f"[{method}] No valid samples.")
        return

    mean_mmd = float(np.mean(all_dist_m))
    os.makedirs(output_dir, exist_ok=True)
    out_txt = os.path.join(output_dir, f"{method}_mMD.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Method: {method}\n")
        f.write("Metric: m-MD (masked Mean Distance, DIFT)\n")
        f.write(f"Num samples: {len(all_dist_m)}\n")
        f.write(f"m-MD: {mean_mmd:.4f}\n")
        if save_per_sample:
            f.write("\nPer-sample:\n")
            for name, mmd in results_per_sample:
                f.write(f"  {name}: {mmd:.4f}\n")
    print(f"[{method}] m-MD: {mean_mmd:.4f} (n={len(all_dist_m)}) -> {out_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute m-MD on DragBench-DR using DIFT and mask from meta_data.pkl."
    )
    parser.add_argument(
        "--source_root", type=str, default=SOURCE_ROOT,
        help="Root of dragbench-dr dataset",
    )
    parser.add_argument(
        "--result_root", type=str, default=RESULT_ROOT,
        help="Root of DragBench-DR results (method subdirs)",
    )
    parser.add_argument(
        "--icrdrag_dir", type=str, default=None,
        help="Flat ICRDrag output dir from inference/run_inference.py.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help="Directory to save m-MD result txt files (default: MD_DragLora)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to evaluate. Default: all 7.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("EVAL_SD_MODEL_PATH") or os.environ.get("EVAL_SD_MODEL_ID"),
        help="SD model path for DIFT",
    )
    parser.add_argument(
        "--default_prompt", type=str, default="a photo",
        help="Default text prompt when meta has no prompt",
    )
    parser.add_argument(
        "--save_per_sample", action="store_true",
        help="Append per-sample m-MD to output txt",
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model_path or EVAL_SD_MODEL_PATH is required for m-MD.")

    methods = ["ICRDrag"] if args.icrdrag_dir else (args.methods if args.methods else METHODS)
    samples = collect_samples(args.source_root)
    if not samples:
        print(f"No samples found under {args.source_root}")
        return

    print(f"Found {len(samples)} samples. Evaluating {len(methods)} methods with DIFT.")

    dift = SDFeaturizer(args.model_path)
    cos = nn.CosineSimilarity(dim=1)

    for method in methods:
        run_mmd_for_method(
            method,
            samples,
            args.result_root,
            args.output_dir,
            dift,
            cos,
            args.default_prompt,
            args.save_per_sample,
            args.icrdrag_dir,
        )

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
