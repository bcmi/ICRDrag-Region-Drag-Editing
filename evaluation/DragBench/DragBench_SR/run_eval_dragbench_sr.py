"""
Evaluate MSE, LPIPS, SSIM for DragBench-SR editing results.

Source images: data/DragBench/openvid_format_dbscan_sr/{sample_id}/original_image.png (flat structure)
Edited images: outputs/dragbench/DragBench-SR/{method}/... (method-specific structure)

Output: one txt file per method under evaluation/DragBench-SR_similarity_results/
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from skimage.metrics import structural_similarity

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data/DragBench/openvid_format_dbscan_sr")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "outputs/dragbench/DragBench-SR")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/evaluation/DragBench/DragBench_SR/similarity")


def preprocess_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize to [-1, 1] and convert to (1, C, H, W)."""
    image = np.asarray(image, dtype=np.float32)
    image = torch.tensor(image, dtype=torch.float32, device=device) / 127.5 - 1
    image = rearrange(image, "h w c -> 1 c h w")
    return image


def collect_source_pairs(source_root: str) -> List[Tuple[str, str]]:
    """
    Walk dragbench-sr (flat structure) and collect (source_path, sample_id)
    for each dir that contains original_image.png.
    Returns list of (source_path, sample_id).
    """
    pairs = []
    for sample_id in sorted(os.listdir(source_root)):
        sample_path = os.path.join(source_root, sample_id)
        if not os.path.isdir(sample_path):
            continue
        orig_path = os.path.join(sample_path, "original_image.png")
        if os.path.isfile(orig_path):
            pairs.append((orig_path, sample_id))
    return pairs


def resolve_edited_path(
    method: str,
    method_root: str,
    sample_id: str,
) -> Optional[str]:
    """
    Resolve edited image path for each method (DragBench-SR flat structure).
    Returns None if file does not exist.
    """
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
    return path if path is not None and os.path.isfile(path) else None


def collect_icrdrag_pairs(icrdrag_root: str) -> List[Tuple[str, str]]:
    """
    Pair images by same filename from ICRDrag/source_image and ICRDrag/edit_image.
    Returns list of (source_path, edited_path).
    """
    source_dir = os.path.join(icrdrag_root, "source_image")
    edit_dir = os.path.join(icrdrag_root, "edit_image")
    if not os.path.isdir(source_dir) or not os.path.isdir(edit_dir):
        return []
    pairs = []
    for name in sorted(os.listdir(edit_dir)):
        edit_path = os.path.join(edit_dir, name)
        if not os.path.isfile(edit_path):
            continue
        source_path = os.path.join(source_dir, name)
        if os.path.isfile(source_path):
            pairs.append((source_path, edit_path))
    return pairs


def collect_icrdrag_flat_pairs(
    icrdrag_dir: str,
    source_pairs: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    pairs = []
    for src_path, sample_id in source_pairs:
        for ext in (".jpg", ".png"):
            edited_path = os.path.join(icrdrag_dir, f"{sample_id}{ext}")
            if os.path.isfile(edited_path):
                pairs.append((src_path, edited_path))
                break
    return pairs


def _resolve_sr_subdir(method_root: str, sample_id: str, filename: str) -> Optional[str]:
    """Find edited image in first subdir that contains sample_id (for DragDiffusion/FastDrag SR)."""
    try:
        subdirs = [d for d in os.listdir(method_root) if os.path.isdir(os.path.join(method_root, d))]
        for sub in sorted(subdirs):
            path = os.path.join(method_root, sub, sample_id, filename)
            if os.path.isfile(path):
                return path
    except OSError:
        pass
    return None


def evaluate_pairs(
    pairs: List[Tuple[str, str]],
    device: torch.device,
    loss_fn: torch.nn.Module,
    mse_loss: torch.nn.Module,
    save_per_pair: bool = False,
    source_root: str = "",
    result_root: str = "",
    edited_panel_index: Optional[int] = None,
) -> Tuple[float, float, float, List[str]]:
    """
    Evaluate MSE, LPIPS, SSIM for (source_path, edited_path) pairs.
    Returns (avg_mse, avg_lpips, avg_ssim, per_pair_lines).
    """
    all_mse = []
    all_lpips = []
    all_ssim = []
    per_pair_lines = []

    for i, (src_path, edt_path) in tqdm(enumerate(pairs), total=len(pairs), desc="Evaluating", unit="pair"):
        source_pil = Image.open(src_path).convert("RGB")
        edited_pil = Image.open(edt_path).convert("RGB")
        if edited_panel_index is not None:
            panel_width = edited_pil.width // 4
            left = edited_panel_index * panel_width
            edited_pil = edited_pil.crop((left, 0, left + panel_width, edited_pil.height))
        edited_pil = edited_pil.resize(source_pil.size, Image.BILINEAR)

        source_np = np.array(source_pil)
        edited_np = np.array(edited_pil)
        source_t = preprocess_image(source_np, device)
        edited_t = preprocess_image(edited_np, device)

        mse = mse_loss(source_t, edited_t)
        all_mse.append(mse.item())

        with torch.no_grad():
            src_224 = F.interpolate(source_t, (224, 224), mode="bilinear")
            edt_224 = F.interpolate(edited_t, (224, 224), mode="bilinear")
            cur_lpips = loss_fn(src_224, edt_224)
            all_lpips.append(cur_lpips.item())

        cur_ssim = structural_similarity(source_np, edited_np, channel_axis=-1, data_range=255)
        all_ssim.append(cur_ssim)

        if save_per_pair:
            src_name = os.path.relpath(src_path, source_root) if source_root else src_path
            edt_name = os.path.relpath(edt_path, result_root) if result_root else edt_path
            per_pair_lines.append(
                f"pair_{i:04d}\tsource={src_name}\tedited={edt_name}\t"
                f"MSE={mse.item():.6f}\tLPIPS={cur_lpips.item():.6f}\tSSIM={cur_ssim:.6f}\n"
            )

    avg_mse = float(np.mean(all_mse))
    avg_lpips = float(np.mean(all_lpips))
    avg_ssim = float(np.mean(all_ssim))
    return avg_mse, avg_lpips, avg_ssim, per_pair_lines


def run_method(
    method: str,
    source_pairs: List[Tuple[str, str]],
    result_root: str,
    device: torch.device,
    loss_fn: torch.nn.Module,
    mse_loss: torch.nn.Module,
    output_dir: str,
    save_per_pair: bool,
    source_root: str,
    icrdrag_dir: Optional[str] = None,
) -> None:
    """Evaluate one method and save results to txt."""
    method_root = os.path.join(result_root, method)
    if method == "ICRDrag" and icrdrag_dir:
        method_root = icrdrag_dir
        pairs = collect_icrdrag_flat_pairs(icrdrag_dir, source_pairs)
        eval_src_root = source_root
        eval_res_root = icrdrag_dir
        edited_panel_index = 2
    elif not os.path.isdir(method_root):
        print(f"[{method}] Skip: directory not found: {method_root}")
        return
    elif method == "ICRDrag":
        # ICRDrag: pair by same filename from source_image and edit_image
        pairs = collect_icrdrag_pairs(method_root)
        eval_src_root = method_root
        eval_res_root = method_root
        edited_panel_index = None
    else:
        pairs = []
        skipped = []
        for src_path, sample_id in source_pairs:
            edt_path = resolve_edited_path(method, method_root, sample_id)
            if edt_path:
                pairs.append((src_path, edt_path))
            else:
                skipped.append(sample_id)
        eval_src_root = source_root
        eval_res_root = result_root
        if skipped:
            print(f"[{method}] Warning: {len(skipped)} samples had no edited image.")
        edited_panel_index = None

    if not pairs:
        print(f"[{method}] No matching pairs.")
        return

    avg_mse, avg_lpips, avg_ssim, per_pair_lines = evaluate_pairs(
        pairs, device, loss_fn, mse_loss, save_per_pair,
        eval_src_root, eval_res_root, edited_panel_index,
    )

    out_path = os.path.join(output_dir, f"{method}_mse_lpips_ssim.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# DragBench-SR Evaluation: {method} ({len(pairs)} pairs)\n")
        f.write(f"# source_root: {eval_src_root}\n")
        f.write(f"# edited_root: {eval_res_root}\n")
        f.write("\n")
        if per_pair_lines:
            f.write("# Per-pair results:\n")
            f.writelines(per_pair_lines)
            f.write("\n")
        f.write("# Summary\n")
        f.write(f"avg_mse={avg_mse:.6f}\n")
        f.write(f"avg_lpips={avg_lpips:.6f}\n")
        f.write(f"avg_ssim={avg_ssim:.6f}\n")
        f.write(f"num_pairs={len(pairs)}\n")

    print(f"[{method}] Saved to {out_path}")
    print(f"  avg_mse={avg_mse:.6f}, avg_lpips={avg_lpips:.6f}, avg_ssim={avg_ssim:.6f}, n={len(pairs)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MSE, LPIPS, SSIM for DragBench-SR methods.")
    parser.add_argument("--source_root", default=SOURCE_ROOT, help="Root of dragbench-sr dataset.")
    parser.add_argument("--result_root", default=RESULT_ROOT, help="Root of DragBench-SR results.")
    parser.add_argument("--icrdrag_dir", default=None,
                        help="Flat ICRDrag output dir from inference/run_inference.py.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory for output txt files.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate. Default: all 7 methods.")
    parser.add_argument("--save_per_pair", action="store_true",
                        help="Write each pair's MSE/LPIPS/SSIM to the txt.")
    args = parser.parse_args()

    default_methods = [
        "DragDiffusion", "DragLoRA", "DragonDiffusion",
        "FastDrag", "GoodDrag", "Inpaint4Drag", "SDE-Drag", "ICRDrag",
    ]
    methods = ["ICRDrag"] if args.icrdrag_dir else (args.methods if args.methods else default_methods)

    source_pairs = collect_source_pairs(args.source_root)
    if not source_pairs:
        raise FileNotFoundError(f"No original_image.png found under {args.source_root}")
    print(f"Found {len(source_pairs)} source samples. Evaluating: {methods}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    mse_loss = torch.nn.MSELoss()

    for method in methods:
        run_method(
            method, source_pairs, args.result_root, device, loss_fn, mse_loss,
            args.output_dir, args.save_per_pair, args.source_root, args.icrdrag_dir,
        )

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
