import os
import re
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from skimage.metrics import structural_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Defaults for the release layout.
REFERENCE_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/PRDBench/test/target_images")
EDITED_IMAGE_DIR = os.path.join(PROJECT_ROOT, "outputs/prdbench/results")
OUTPUT_TXT = os.path.join(PROJECT_ROOT, "outputs/evaluation/similarity_results/icrdrag_results.txt")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize to [-1, 1] and convert to (1, C, H, W)."""
    image = np.asarray(image, dtype=np.float32)
    image = torch.tensor(image, dtype=torch.float32, device=device) / 127.5 - 1
    image = rearrange(image, "h w c -> 1 c h w")
    return image


def get_sorted_image_files(dir_path: str) -> list:
    """Return sorted list of image filenames in dir_path."""
    names = [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        and not f.startswith(".")
    ]
    return sorted(names)


def get_source_basename(source_filename: str) -> str:
    """Return source basename without extension."""
    return os.path.splitext(source_filename)[0]


def get_video_identifier(source_basename: str) -> str:
    """
    Get video identifier: remove trailing _<digit>_src (e.g. _0_src).
    E.g. _0am5dxWtsg_32_0to109_0_src -> _0am5dxWtsg_32_0to109
    """
    match = re.match(r"^(.+)_\d+_tgt$", source_basename)
    return match.group(1) if match else source_basename


def find_edited_by_basename(source_basename: str, video_id: str, edited_names: list) -> Optional[str]:
    """
    Find edited image whose filename contains source_basename.
    If none, try video_id (for FastDrag-style: 00000_-xOWEgcB-cA_10_0to105_frame0_to_1.png).
    Prefer exact match, else first by sorted order.
    """
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MSE, LPIPS and SSIM for reference vs edited image pairs.")
    parser.add_argument("--source_dir", default=REFERENCE_IMAGE_DIR, help="Directory of reference images.")
    parser.add_argument("--edited_dir", default=EDITED_IMAGE_DIR, help="Directory of edited images.")
    parser.add_argument("--output", default=OUTPUT_TXT, help="Output txt file for results.")
    parser.add_argument("--save_per_pair", action="store_true", help="If set, write each pair's MSE/LPIPS/SSIM to the txt.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    mse_loss = torch.nn.MSELoss()

    source_names = get_sorted_image_files(args.source_dir)
    edited_names = get_sorted_image_files(args.edited_dir)

    n_source = len(source_names)
    n_edited = len(edited_names)
    if n_source == 0 or n_edited == 0:
        raise FileNotFoundError(f"No images found: source_dir has {n_source} images, edited_dir has {n_edited}.")

    # Build pairs: for each source, use basename; find edited whose filename contains basename (or video_id)
    pairs = []
    skipped = []
    for src_name in source_names:
        src_basename = get_source_basename(src_name)
        video_id = get_video_identifier(src_basename)
        edt_name = find_edited_by_basename(src_basename, video_id, edited_names)
        if edt_name is not None:
            pairs.append((src_name, edt_name))
        else:
            skipped.append((src_name, src_basename))

    n_pairs = len(pairs)
    if n_pairs == 0:
        raise FileNotFoundError(
            f"No image pairs matched by basename. source_dir has {n_source} images, "
            f"edited_dir has {n_edited}. Check that edited filenames contain the source basename "
            f"or video identifier (e.g. _0am5dxWtsg_32_0to109)."
        )
    if skipped:
        print(f"Warning: {len(skipped)} source images had no matching edited image (skipped).")

    all_mse = []
    all_lpips = []
    all_ssim = []
    per_pair_lines = []

    for i, (src_name, edt_name) in tqdm(enumerate(pairs), total=n_pairs, desc="Evaluating", unit="pair"):
        src_path = os.path.join(args.source_dir, src_name)
        edt_path = os.path.join(args.edited_dir, edt_name)

        source_pil = Image.open(src_path).convert("RGB")
        edited_pil = Image.open(edt_path).convert("RGB")
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
            cur_lpips = loss_fn_alex(src_224, edt_224)
            all_lpips.append(cur_lpips.item())

        # SSIM on [0, 255] RGB images
        cur_ssim = structural_similarity(source_np, edited_np, channel_axis=-1, data_range=255)
        all_ssim.append(cur_ssim)

        if args.save_per_pair:
            per_pair_lines.append(
                f"pair_{i:04d}\tsource={src_name}\tedited={edt_name}\t"
                f"MSE={mse.item():.6f}\tLPIPS={cur_lpips.item():.6f}\tSSIM={cur_ssim:.6f}\n"
            )

    avg_mse = np.mean(all_mse)
    avg_lpips = np.mean(all_lpips)
    avg_ssim = np.mean(all_ssim)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation: source vs edited ({n_pairs} pairs)\n")
        f.write(f"# source_dir: {args.source_dir}\n")
        f.write(f"# edited_dir: {args.edited_dir}\n")
        f.write("\n")
        if per_pair_lines:
            f.write("# Per-pair results:\n")
            f.writelines(per_pair_lines)
            f.write("\n")
        f.write("# Summary\n")
        f.write(f"avg_mse={avg_mse:.6f}\n")
        f.write(f"avg_lpips={avg_lpips:.6f}\n")
        f.write(f"avg_ssim={avg_ssim:.6f}\n")
        f.write(f"num_pairs={n_pairs}\n")

    print(f"Results saved to {args.output}")
    print(f"avg_mse: {avg_mse:.6f}")
    print(f"avg_lpips: {avg_lpips:.6f}")
    print(f"avg_ssim: {avg_ssim:.6f}")
    print(f"num_pairs: {n_pairs}")


if __name__ == "__main__":
    main()
