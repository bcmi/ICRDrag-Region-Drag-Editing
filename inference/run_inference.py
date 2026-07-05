import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ICRDrag.diffusion.pipelines.icrdrag import ICRDragPipeline
from ICRDrag.models.denoiser.nextdit import NextDiT


PROMPT = "[[semanticdrag]] regional editing"
NEGATIVE_PROMPT = (
    "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, "
    "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, "
    "poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
)


def image2cv2(image):
    image_np = np.array(image)
    if image_np.shape[-1] == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if image_np.shape[-1] == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    return image_np


def normalize_testset(name):
    key = name.lower().replace("_", "-")
    aliases = {
        "prd": "prdbench",
        "prdbench": "prdbench",
        "prdb": "prdbench",
        "dragbench": "dragbench-dr",
        "dragbench-dr": "dragbench-dr",
        "dragbench-sr": "dragbench-sr",
        "db-dr": "dragbench-dr",
        "db-sr": "dragbench-sr",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported testset: {name}. Use prdbench, dragbench, dragbench-dr, or dragbench-sr."
        )
    return aliases[key]


def iter_dragbench_dr(root_dir, valid_names=None):
    for category in sorted(os.listdir(root_dir)):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        for item in sorted(os.listdir(category_path)):
            if valid_names is not None and item not in valid_names:
                continue
            item_path = os.path.join(category_path, item)
            if os.path.isdir(item_path):
                yield item, item_path


def iter_dragbench_sr(root_dir):
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            yield item, item_path


def load_valid_names(valid_list_dir):
    if valid_list_dir is None:
        return None
    valid_dir = Path(valid_list_dir)
    if not valid_dir.is_dir():
        raise FileNotFoundError(f"valid_list_dir does not exist: {valid_dir}")
    return {Path(name).stem for name in os.listdir(valid_dir)}


def load_pipeline(args):
    device = torch.device(args.device)
    pipeline = ICRDragPipeline.from_pretrained(args.model_path).to(
        device=device, dtype=torch.bfloat16
    )
    if args.ckpt_folder:
        transformer = NextDiT.from_pretrained(args.ckpt_folder, subfolder=args.ckpt_number).to(
            device=device, dtype=torch.bfloat16
        )
        pipeline.transformer = transformer
    return pipeline, device


def run_icrdrag(pipeline, image, num_inference_steps, guidance_scale):
    """Run the shared ICRDrag img2img call used by PRDBench and DragBench."""
    return pipeline.img2img(
        image=image,
        num_inference_steps=num_inference_steps,
        prompt=PROMPT,
        denoise_mask=[0, 0, 1, 0],
        guidance_scale=guidance_scale,
        negative_prompt=NEGATIVE_PROMPT,
    ).images[0]


def load_rgb_image(path, image_width, image_height):
    return Image.open(path).convert("RGB").resize((image_width, image_height))


def load_prd_sample(entry, base_path, image_width, image_height):
    src_img_name = os.path.basename(entry["src_img_path"])
    src_mask_name = os.path.basename(entry["src_mask_path"]).replace("mask", "color")
    tgt_mask_name = os.path.basename(entry["tgt_mask_path"]).replace("mask", "color")

    src_img_path = os.path.join(base_path, "source_images", src_img_name)
    src_mask_path = os.path.join(base_path, "source_masks_color", src_mask_name)
    tgt_mask_path = os.path.join(base_path, "target_masks_color", tgt_mask_name)

    start_frame = load_rgb_image(src_img_path, image_width, image_height)
    start_mask = load_rgb_image(src_mask_path, image_width, image_height)
    target_mask = load_rgb_image(tgt_mask_path, image_width, image_height)

    return {
        "name": entry["video_name"],
        "start_image": start_frame,
        "source_mask": start_mask,
        "target_mask": target_mask,
    }


def run_prdbench(args, pipeline):
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.drag_data_json, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} samples from {args.drag_data_json}")

    end_index = len(data_list)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + args.max_samples)

    for i in tqdm(range(args.start_index, end_index), desc="PRDBench inference"):
        sample = load_prd_sample(
            data_list[i], args.base_path, args.image_width, args.image_height
        )

        result_path = os.path.join(results_dir, f"{i}_{sample['name']}_result.png")
        result_img = run_icrdrag(
            pipeline,
            [sample["start_image"], sample["source_mask"], sample["target_mask"]],
            args.num_inference_steps,
            args.guidance_scale,
        )
        result_img.save(result_path)


def run_dragbench(args, pipeline, split):
    os.makedirs(args.output_dir, exist_ok=True)
    valid_names = load_valid_names(args.valid_list_dir)
    iterator = (
        iter_dragbench_sr(args.root_dir)
        if split == "sr"
        else iter_dragbench_dr(args.root_dir, valid_names)
    )

    required_files = {"original_image.png", "source_mask.png", "target_mask.png"}
    samples = list(iterator)
    if args.max_samples is not None:
        samples = samples[args.start_index : args.start_index + args.max_samples]
    elif args.start_index:
        samples = samples[args.start_index :]

    for item, item_path in tqdm(samples, desc=f"DragBench-{split.upper()} inference"):
        filenames = set(os.listdir(item_path))
        if not required_files.issubset(filenames):
            continue

        output_path = os.path.join(args.output_dir, f"{item}.jpg")
        if os.path.exists(output_path):
            continue

        original_image = Image.open(os.path.join(item_path, "original_image.png")).convert("RGB")
        source_mask = Image.open(os.path.join(item_path, "source_mask.png")).convert("RGB")
        target_mask = Image.open(os.path.join(item_path, "target_mask.png")).convert("RGB")

        result_img = run_icrdrag(
            pipeline,
            [original_image, source_mask, target_mask],
            args.num_inference_steps,
            args.guidance_scale,
        )

        start_img_cv2 = image2cv2(original_image)
        source_mask_cv2 = image2cv2(source_mask)
        target_mask_cv2 = image2cv2(target_mask)
        edited_img_cv2 = image2cv2(result_img)
        edited_img_cv2 = cv2.resize(edited_img_cv2, start_img_cv2.shape[:2][::-1])
        # DragBench release outputs use the legacy four-panel format expected by the evaluators.
        cv2.imwrite(
            output_path,
            np.concatenate(
                [start_img_cv2, source_mask_cv2, edited_img_cv2, target_mask_cv2],
                axis=1,
            ),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ICRDrag inference on PRDBench or DragBench with one entrypoint."
    )
    parser.add_argument(
        "--testset",
        default="prdbench",
        help="Inference testset: prdbench, dragbench, dragbench-dr, or dragbench-sr.",
    )
    parser.add_argument(
        "--root_dir",
        default=None,
        help="DragBench root directory. Defaults depend on --testset.",
    )
    parser.add_argument(
        "--drag_data_json",
        default=str(PROJECT_ROOT / "data/PRDBench/test/drag_data.json"),
        help="Path to PRDBench drag_data.json.",
    )
    parser.add_argument(
        "--base_path",
        default=str(PROJECT_ROOT / "data/PRDBench/test"),
        help="PRDBench root containing source_images, target_images and mask folders.",
    )
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "weights/ICRDrag"),
        help="ICRDrag pipeline directory.",
    )
    parser.add_argument(
        "--ckpt_folder",
        default=None,
        help="Optional transformer checkpoint folder. If omitted, the transformer in model_path is used.",
    )
    parser.add_argument("--ckpt_number", default="transformer", help="Checkpoint subfolder name.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults depend on --testset.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--valid_list_dir", default=None, help="Optional DragBench-DR allow-list dir.")

    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    testset = normalize_testset(args.testset)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.root_dir is None:
        if testset == "dragbench-sr":
            args.root_dir = str(PROJECT_ROOT / "data/DragBench/openvid_format_dbscan_sr")
        else:
            args.root_dir = str(PROJECT_ROOT / "data/DragBench/openvid_format_dbscan_dr")

    if args.output_dir is None:
        defaults = {
            "prdbench": PROJECT_ROOT / "outputs/prdbench",
            "dragbench-dr": PROJECT_ROOT / "outputs/dragbench/DragBench-DR",
            "dragbench-sr": PROJECT_ROOT / "outputs/dragbench/DragBench-SR",
        }
        args.output_dir = str(defaults[testset])

    pipeline, _ = load_pipeline(args)

    if testset == "prdbench":
        run_prdbench(args, pipeline)
    elif testset == "dragbench-sr":
        run_dragbench(args, pipeline, split="sr")
    else:
        run_dragbench(args, pipeline, split="dr")


if __name__ == "__main__":
    main()
