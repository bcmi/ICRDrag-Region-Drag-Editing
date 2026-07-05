import os
import sys
import re
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm
from typing import Optional, List, Tuple

# Add DragLoRA/drag_bench_evaluation for DIFT when provided; otherwise use bundled dift_sd.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIRD_PARTY_PATH = os.path.join(PROJECT_ROOT, "evaluation", "third_party")
if THIRD_PARTY_PATH not in sys.path:
    sys.path.insert(0, THIRD_PARTY_PATH)
DRAGLORA_EVAL_PATH = os.path.join(PROJECT_ROOT, "DragLoRA", "drag_bench_evaluation")
if os.path.isdir(DRAGLORA_EVAL_PATH) and DRAGLORA_EVAL_PATH not in sys.path:
    sys.path.insert(0, DRAGLORA_EVAL_PATH)

from dift_sd import SDFeaturizer  # type: ignore[import-not-found]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_sorted_image_files(dir_path: str) -> list:
    """Return sorted list of image filenames in dir_path."""
    names = [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS and not f.startswith(".")
    ]
    return sorted(names)


def get_source_basename(source_filename: str) -> str:
    """Return source basename without extension."""
    return os.path.splitext(source_filename)[0]


def get_video_identifier(source_basename: str) -> str:
    """Remove trailing _<digit>_src. E.g. 'xxx_0_src' -> 'xxx'."""
    match = re.match(r"^(.+)_\d+_src$", source_basename)
    return match.group(1) if match else source_basename


def find_edited_by_basename(
    source_basename: str, video_id: str, edited_names: list
) -> Optional[str]:
    """Find edited image whose filename contains source_basename or video_id."""
    for key in (source_basename, video_id):
        candidates = []
        for f in edited_names:
            base_no_ext = os.path.splitext(f)[0]
            if key in base_no_ext:
                candidates.append((base_no_ext == key, f))
        if candidates:
            candidates.sort(key=lambda x: (not x[0], x[1]))
            return candidates[0][1]
    return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_mask(
    src_mask_path: str,
    tgt_mask_path: str,
    target_size: Tuple[int, int],
    mode: str = "target",
    white_threshold: int = 127,
) -> np.ndarray:
    """
    Load PRD masks and resize to target_size, returning one binary mask.
    Binary masks use white = mask. Color masks use neutral gray as background
    and non-gray colored regions as mask.
    DragLoRA's original m-MD evaluator uses one mask in the edited image; PRD target masks
    are the closest equivalent.
    """
    def load_binary_mask(path: str, size: Tuple[int, int]) -> np.ndarray:
        if not os.path.isfile(path):
            return np.zeros(size, dtype=np.uint8)
        img = np.array(Image.open(path))
        if img.ndim == 3:
            channel_delta = np.max(img, axis=2) - np.min(img, axis=2)
            if channel_delta.max() <= 5:
                img = img[:, :, 0]
                mask = (img > white_threshold).astype(np.uint8)
            else:
                mask = (np.max(np.abs(img.astype(np.int16) - 128), axis=2) > 5).astype(np.uint8)
        else:
            mask = (img > white_threshold).astype(np.uint8)
        if mask.shape[:2] != size:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask)
            mask_pil = mask_pil.resize((size[1], size[0]), PILImage.NEAREST)
            mask = np.array(mask_pil)
        return mask

    h, w = target_size
    if mode == "source":
        return load_binary_mask(src_mask_path, (h, w))
    if mode == "target":
        return load_binary_mask(tgt_mask_path, (h, w))
    if mode == "union":
        m_src = load_binary_mask(src_mask_path, (h, w))
        m_tgt = load_binary_mask(tgt_mask_path, (h, w))
        return np.clip(m_src + m_tgt, 0, 1).astype(np.uint8)
    raise ValueError(f"Unsupported mask mode: {mode}")


def points_from_entry(entry: dict, H: int, W: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Get handle_points and target_points in (y, x) order, clamped to image bounds.
    entry['points']: list of {handle_point: [x,y], target_point: [x,y]}.
    """
    handle_points = []
    target_points = []
    for p in entry["points"]:
        hx, hy = float(p["handle_point"][0]), float(p["handle_point"][1])
        tx, ty = float(p["target_point"][0]), float(p["target_point"][1])
        hy_int = int(round(np.clip(hy, 0, H - 1)))
        hx_int = int(round(np.clip(hx, 0, W - 1)))
        ty_int = int(round(np.clip(ty, 0, H - 1)))
        tx_int = int(round(np.clip(tx, 0, W - 1)))
        handle_points.append(torch.tensor([hy_int, hx_int]))
        target_points.append(torch.tensor([ty_int, tx_int]))
    return handle_points, target_points


def main():
    parser = argparse.ArgumentParser(
        description="Compute m-MD on PRDBench using DragLoRA-style DIFT and a single search mask."
    )
    parser.add_argument(
        "--drag_data_json",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/PRDBench/test/drag_data.json"
        ),
        help="Path to drag_data.json",
    )
    parser.add_argument(
        "--source_images_dir",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/PRDBench/test/source_images"
        ),
        help="Directory of source images",
    )
    parser.add_argument(
        "--edited_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs/prdbench/results"),
        help="Directory of edited images (e.g. GoodDrag)",
    )
    parser.add_argument(
        "--source_masks_dir",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/PRDBench/test/source_masks_color"
        ),
        help="Directory of source masks",
    )
    parser.add_argument(
        "--target_masks_dir",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/PRDBench/test/target_masks_color"
        ),
        help="Directory of target masks",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("EVAL_SD_MODEL_PATH") or os.environ.get("EVAL_SD_MODEL_ID"),
        help="Stable Diffusion model path for DIFT",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs/evaluation/mMD_results/icrdrag_mMD.txt"),
        help="Output txt path for m-MD summary",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="ICRDrag",
        help="Method name for output file",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default="",
        help="Default text prompt for DIFT (dataset has no per-image prompt)",
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="target",
        choices=["target", "source", "union"],
        help="Which PRD mask to use as DragLoRA's single m-MD search mask",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for DIFT feature extraction, matching DragLoRA evaluation",
    )
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="If set, append per-sample m-MD to output txt",
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model_path or EVAL_SD_MODEL_PATH is required for m-MD.")

    seed_everything(args.seed)
    edited_names = get_sorted_image_files(args.edited_dir)
    if not edited_names:
        raise FileNotFoundError(f"No images in edited_dir: {args.edited_dir}")

    with open(args.drag_data_json, "r") as f:
        drag_data = json.load(f)

    # Resolve mask path: use absolute path from entry if exists, else under mask dir
    def resolve_mask_path(ent, path_key: str, use_src_dir: bool) -> str:
        abspath = ent.get(path_key, "")
        if abspath and os.path.isfile(abspath):
            return abspath
        basename = os.path.basename(abspath) if abspath else ""
        dir_path = args.source_masks_dir if use_src_dir else args.target_masks_dir
        return os.path.join(dir_path, basename)

    dift = SDFeaturizer(args.model_path)
    cos = nn.CosineSimilarity(dim=1)
    all_dist_m = []
    results_per_sample = []

    for entry in tqdm(drag_data, desc="Computing m-MD"):
        src_img_path = entry["src_img_path"]
        if not os.path.isfile(src_img_path):
            src_basename = os.path.basename(src_img_path)
            src_img_path = os.path.join(args.source_images_dir, src_basename)
        if not os.path.isfile(src_img_path):
            continue

        src_basename = os.path.basename(src_img_path)
        source_basename = get_source_basename(src_basename)
        video_id = get_video_identifier(source_basename)
        edt_name = find_edited_by_basename(source_basename, video_id, edited_names)
        if edt_name is None:
            continue

        edited_path = os.path.join(args.edited_dir, edt_name)
        source_pil = Image.open(src_img_path).convert("RGB")
        edited_pil = Image.open(edited_path).convert("RGB")
        edited_pil = edited_pil.resize(source_pil.size, Image.BILINEAR)
        H, W = source_pil.size[1], source_pil.size[0]

        src_mask_path = resolve_mask_path(entry, "src_mask_path", use_src_dir=True)
        tgt_mask_path = resolve_mask_path(entry, "tgt_mask_path", use_src_dir=False)
        mask_np = load_eval_mask(src_mask_path, tgt_mask_path, (H, W), mode=args.mask_mode)
        mask = torch.from_numpy(mask_np)
        mask_flat = mask.flatten()

        handle_points, target_points = points_from_entry(entry, H, W)
        if not handle_points:
            continue

        # Image tensors: (1, 3, H, W), range [-1, 1]
        source_tensor = (PILToTensor()(source_pil) / 255.0 - 0.5) * 2
        if source_tensor.dim() == 3:
            source_tensor = source_tensor.unsqueeze(0)
        dragged_tensor = (PILToTensor()(edited_pil) / 255.0 - 0.5) * 2
        if dragged_tensor.dim() == 3:
            dragged_tensor = dragged_tensor.unsqueeze(0)
        _, _, h_in, w_in = source_tensor.shape
        if (h_in, w_in) != (H, W):
            source_tensor = F.interpolate(
                source_tensor, (H, W), mode="bilinear"
            )
            dragged_tensor = F.interpolate(
                dragged_tensor, (H, W), mode="bilinear"
            )

        with torch.no_grad():
            ft_source = dift.forward(
                source_tensor,
                prompt=args.default_prompt,
                t=261,
                up_ft_index=1,
                ensemble_size=8,
            )
            ft_source = F.interpolate(ft_source, (H, W), mode="bilinear")
            ft_dragged = dift.forward(
                dragged_tensor,
                prompt=args.default_prompt,
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
            hp = handle_points[pt_idx]
            tp = target_points[pt_idx]
            num_channel = ft_source.size(1)
            src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
            cos_map = cos(src_vec.squeeze(-1), ft_dragged_mask)[0]
            max_rc_m = candidate_coords[cos_map.argmax()]
            dist_m = (tp - max_rc_m.float()).norm()
            sample_dists.append(dist_m.item())
            all_dist_m.append(dist_m.item())
        sample_mmd = np.mean(sample_dists)
        sample_id = f"{video_id}_{entry.get('frame_idx_source', '')}_{entry.get('frame_idx_target', '')}"
        results_per_sample.append((sample_id, sample_mmd))

    if not all_dist_m:
        print("No valid samples to compute m-MD.")
        return

    mean_mmd = float(np.mean(all_dist_m))
    out_dir = os.path.dirname(os.path.abspath(args.output_txt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(f"Method: {args.method_name}\n")
        f.write("Metric: m-MD (masked Mean Distance, DragLoRA-style DIFT)\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Mask mode: {args.mask_mode}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Default prompt: {args.default_prompt!r}\n")
        f.write(f"Num samples: {len(results_per_sample)}\n")
        f.write(f"Num points: {len(all_dist_m)}\n")
        f.write(f"m-MD: {mean_mmd:.4f}\n")
        if args.save_per_sample:
            f.write("\nPer-sample:\n")
            for sid, mmd in results_per_sample:
                f.write(f"  {sid}: {mmd:.4f}\n")
    print(f"m-MD: {mean_mmd:.4f} (n_samples={len(results_per_sample)}, n_points={len(all_dist_m)})")
    print(f"Saved to {args.output_txt}")


if __name__ == "__main__":
    main()
