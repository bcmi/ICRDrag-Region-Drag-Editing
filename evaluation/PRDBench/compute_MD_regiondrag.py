import os
import sys
import re
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add bundled RegionDrag utilities for DragEvaluator.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGION_DRAG_PATH = os.path.join(PROJECT_ROOT, "evaluation", "third_party")
if REGION_DRAG_PATH not in sys.path:
    sys.path.insert(0, REGION_DRAG_PATH)

from region_utils.evaluator import DragEvaluator


def get_source_basename(filename):
    """Source basename without extension. E.g. '_0am5dxWtsg_32_0to109_0_src.jpg' -> '_0am5dxWtsg_32_0to109_0_src'."""
    return os.path.splitext(filename)[0]


def get_video_identifier(source_basename):
    """Remove trailing _<number>_src. E.g. '_0am5dxWtsg_32_0to109_0_src' -> '_0am5dxWtsg_32_0to109'."""
    match = re.match(r"^(.+)_\d+_src$", source_basename)
    return match.group(1) if match else source_basename


def find_edited_image(edited_dir, source_basename, video_id):
    """
    与 run_eval_mse_lpips_ssim 一致：先用 source_basename 再 video_id 在 edited 中找「包含」该 key 的文件；
    有多个时优先取 basename 精确等于 key 的，否则按排序取第一个。
    """
    for key in (source_basename, video_id):
        candidates = []
        for f in os.listdir(edited_dir):
            path = os.path.join(edited_dir, f)
            if not os.path.isfile(path):
                continue
            base_no_ext = os.path.splitext(f)[0]
            if key in base_no_ext:
                candidates.append((base_no_ext == key, f))
        if candidates:
            candidates.sort(key=lambda x: (not x[0], x[1]))
            return os.path.join(edited_dir, candidates[0][1])
    return None


def load_image(path):
    """与 get_drag_data 一致：PIL 读图后 np.array。"""
    return np.array(Image.open(path))


def main():
    parser = argparse.ArgumentParser(description="Compute MD on new dataset for a method (e.g. GoodDrag).")
    parser.add_argument("--drag_data_json", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/PRDBench/test/drag_data.json"),
                        help="Path to drag_data.json")
    parser.add_argument("--source_images_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/PRDBench/test/source_images_rgb"),
                        help="Directory of source images")
    parser.add_argument("--edited_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/prdbench/results"),
                        help="Directory of edited images (e.g. GoodDrag)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/evaluation/MD_results"),
                        help="Directory to save MD result txt")
    parser.add_argument("--method_name", type=str, default="ICRDrag",
                        help="Name of the method (used for output filename)")
    parser.add_argument("--sd_method", type=str, default="dino", choices=["sd", "dino"],
                        help="Feature method for MD: sd or dino")
    parser.add_argument("--sd_model_path", type=str,
                        default=os.environ.get("EVAL_SD_MODEL_PATH") or os.environ.get("EVAL_SD_MODEL_ID"),
                        help="Stable Diffusion model id or local path used when --sd_method sd")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="If set, save per-sample MD visualization: source+target with src_pts on source, pred_trg_pts and GT trg_pts on target")
    parser.add_argument("--vis_max_samples", type=int, default=None,
                        help="Max samples to visualize when --vis_dir set (default: all)")
    args = parser.parse_args()
    if args.sd_method == "sd" and not args.sd_model_path:
        parser.error("--sd_model_path or EVAL_SD_MODEL_PATH is required when --sd_method sd.")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for batch saving

    with open(args.drag_data_json, "r") as f:
        drag_data = json.load(f)

    evaluator = DragEvaluator(sd_model_path=args.sd_model_path)
    all_distances = []
    results_per_sample = []
    vis_count = 0

    for entry in tqdm(drag_data, desc="Computing MD"):
        src_img_path = entry["src_img_path"]
        if not os.path.isfile(src_img_path):
            # Try under source_images_dir with basename
            src_basename = os.path.basename(src_img_path)
            src_img_path = os.path.join(args.source_images_dir, src_basename)
        if not os.path.isfile(src_img_path):
            continue

        src_basename = os.path.basename(src_img_path)
        source_basename = get_source_basename(src_basename)
        video_id = get_video_identifier(source_basename)
        frame_idx_source = entry["frame_idx_source"]
        frame_idx_target = entry["frame_idx_target"]

        edited_path = find_edited_image(args.edited_dir, source_basename, video_id)
        if edited_path is None:
            continue

        ori_image = load_image(src_img_path)
        out_image = load_image(edited_path)

        points = entry["points"]
        source_pts = [[int(p["handle_point"][0]), int(p["handle_point"][1])] for p in points]
        target_pts = [[int(p["target_point"][0]), int(p["target_point"][1])] for p in points]

        if len(source_pts) == 0:
            continue

        prompt = ""  # empty as requested
        sample_id = f"{video_id}_{frame_idx_source}_{frame_idx_target}"
        plot_path = None
        if args.vis_dir and (args.vis_max_samples is None or vis_count < args.vis_max_samples):
            plot_path = os.path.join(args.vis_dir, f"{sample_id.replace('/', '_')}.jpg")
        try:
            md = evaluator.compute_distance(
                ori_image, out_image,
                source_pts, target_pts,
                method=args.sd_method,
                prompt=prompt if args.sd_method == "sd" else None,
                plot_path=plot_path,
            )
        except Exception as e:
            print(f"Skip {video_id} {frame_idx_source}_{frame_idx_target}: {e}")
            continue

        all_distances.append(md)
        results_per_sample.append((sample_id, md))
        if plot_path:
            vis_count += 1

    if not all_distances:
        print("No valid samples to compute MD.")
        return

    mean_md = torch.tensor(all_distances).mean().item()
    out_txt = os.path.join(args.output_dir, f"{args.method_name}_MD.txt")
    with open(out_txt, "w") as f:
        f.write(f"Method: {args.method_name}\n")
        f.write(f"Feature: {args.sd_method}\n")
        f.write(f"Num samples: {len(all_distances)}\n")
        f.write(f"MD: {mean_md:.4f}\n")
        f.write("\nPer-sample:\n")
        for name, md in results_per_sample:
            f.write(f"  {name}: {md:.4f}\n")
    print(f"MD: {mean_md:.4f} (n={len(all_distances)})")
    print(f"Saved to {out_txt}")
    if args.vis_dir:
        print(f"Saved {vis_count} visualizations to {args.vis_dir}")


if __name__ == "__main__":
    main()
