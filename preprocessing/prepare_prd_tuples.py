import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
cv2 = None
Image = None

PALETTE = np.array(
    [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
        [230, 85, 13],
        [49, 130, 189],
        [117, 107, 177],
    ],
    dtype=np.uint8,
)


def load_image_dependencies():
    global cv2, Image
    if cv2 is not None and Image is not None:
        return
    import cv2 as cv2_module
    from PIL import Image as pil_image

    cv2 = cv2_module
    Image = pil_image


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_video_path(video_path, video_root, jsonl_path):
    path = Path(video_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = []
    if video_root is not None:
        candidates.extend([video_root / path, video_root / path.name])
    candidates.append(jsonl_path.parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def load_frame_pair(video_path, start_idx, image_size, video_channel_order):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    wanted = {start_idx, start_idx + 1}
    idx = 0
    while cap.isOpened() and len(frames) < 2:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            if video_channel_order == "bgr":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()

    if len(frames) != 2:
        raise RuntimeError(f"Cannot read adjacent frames {start_idx}/{start_idx + 1}: {video_path}")
    return frames[0], frames[1]


def load_label_pair(mask_path, start_idx, image_size):
    masks = np.load(mask_path, mmap_mode="r")
    if masks.ndim == 4:
        masks = masks[:, 0]
    if start_idx + 1 >= masks.shape[0]:
        raise RuntimeError(f"Mask has too few frames for index {start_idx}: {mask_path}")

    src = cv2.resize(
        np.asarray(masks[start_idx]),
        image_size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)
    tgt = cv2.resize(
        np.asarray(masks[start_idx + 1]),
        image_size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)
    return src, tgt


def select_labels(src_labels, tgt_labels, rng, max_regions, min_area, include_background):
    src_unique = set(np.unique(src_labels).tolist())
    tgt_unique = set(np.unique(tgt_labels).tolist())
    labels = sorted(src_unique & tgt_unique)
    if not include_background:
        labels = [label for label in labels if label != 0]

    valid = []
    for label in labels:
        if (src_labels == label).sum() >= min_area and (tgt_labels == label).sum() >= min_area:
            valid.append(label)

    if not valid:
        return []

    count = min(rng.integers(1, max_regions + 1), len(valid))
    chosen = rng.choice(valid, size=count, replace=False)
    return [int(label) for label in np.sort(chosen)]


def make_region_mask(labels, selected_labels, dilation_kernel):
    mask = np.full((*labels.shape, 3), 128, dtype=np.uint8)
    kernel = None
    if dilation_kernel > 0:
        kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)

    for idx, label in enumerate(selected_labels):
        region = (labels == label).astype(np.uint8)
        if kernel is not None:
            region = cv2.dilate(region, kernel, iterations=1)
        mask[region.astype(bool)] = PALETTE[idx % len(PALETTE)]
    return mask


def save_rgb(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def build_tuple(record, args, rng, tuple_index):
    video_path = resolve_video_path(record["video_path"], args.video_root, args.video_jsonl)
    name = video_path.stem
    mask_path = args.mask_root / f"{name}.npy"
    if not mask_path.exists():
        raise RuntimeError(f"Missing mask: {mask_path}")

    mask_meta = np.load(mask_path, mmap_mode="r")
    num_frames = int(mask_meta.shape[0])
    if num_frames < 2:
        raise RuntimeError(f"Need at least two mask frames: {mask_path}")

    start_idx = int(rng.integers(0, num_frames - 1))
    source_image, target_image = load_frame_pair(
        video_path, start_idx, args.image_size, args.video_channel_order
    )
    source_labels, target_labels = load_label_pair(mask_path, start_idx, args.image_size)
    selected_labels = select_labels(
        source_labels,
        target_labels,
        rng,
        args.max_regions,
        args.min_region_area,
        args.include_background_label,
    )
    if not selected_labels:
        raise RuntimeError(f"No valid shared region labels: {name}")

    source_region_mask = make_region_mask(source_labels, selected_labels, args.dilation_kernel_size)
    target_region_mask = make_region_mask(target_labels, selected_labels, args.dilation_kernel_size)

    sample_id = f"{tuple_index:06d}_{name}_{start_idx}_to_{start_idx + 1}"
    paths = {
        "source_image": Path("source_images") / f"{sample_id}_src.png",
        "target_image": Path("target_images") / f"{sample_id}_tgt.png",
        "source_region_mask": Path("source_region_masks") / f"{sample_id}_src_mask.png",
        "target_region_mask": Path("target_region_masks") / f"{sample_id}_tgt_mask.png",
    }

    save_rgb(args.output_dir / paths["source_image"], source_image)
    save_rgb(args.output_dir / paths["target_image"], target_image)
    save_rgb(args.output_dir / paths["source_region_mask"], source_region_mask)
    save_rgb(args.output_dir / paths["target_region_mask"], target_region_mask)

    return {
        "id": sample_id,
        "video_name": name,
        "frame_idx_source": start_idx,
        "frame_idx_target": start_idx + 1,
        "selected_region_labels": selected_labels,
        "source_image": str(paths["source_image"]),
        "source_region_mask": str(paths["source_region_mask"]),
        "target_image": str(paths["target_image"]),
        "target_region_mask": str(paths["target_region_mask"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct PRD-style source/target image and region-mask tuples from "
            "time-subsampled videos and per-frame panoptic segmentation masks."
        )
    )
    parser.add_argument(
        "--video_jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/PRDBench/train/openvid_jsonl/train.jsonl",
    )
    parser.add_argument(
        "--video_root",
        type=Path,
        default=PROJECT_ROOT / "data/PRD/train/processed_videos_rgb",
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=PROJECT_ROOT / "data/PRD/train/mask_numpys",
    )
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs/prd_tuples")
    parser.add_argument("--num_tuples", type=int, default=None)
    parser.add_argument("--samples_per_video", type=int, default=1)
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Maximum tuple attempts when --num_tuples is set. Defaults to num_tuples * 20.",
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=(512, 512), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--video_channel_order",
        choices=("rgb", "bgr"),
        default="bgr",
        help=(
            "Channel order returned by cv2.VideoCapture. Use bgr for the converted "
            "processed_videos_rgb release, or rgb for the original processed_videos."
        ),
    )
    parser.add_argument("--max_regions", type=int, default=5)
    parser.add_argument("--min_region_area", type=int, default=64)
    parser.add_argument("--dilation_kernel_size", type=int, default=15)
    parser.add_argument("--include_background_label", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    load_image_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    records = list(iter_jsonl(args.video_jsonl))
    if not records:
        raise RuntimeError(f"No records found in {args.video_jsonl}")

    metadata = []
    skipped = 0

    if args.num_tuples is None:
        pbar = tqdm(records, desc="Preparing PRD tuples")
        for record in pbar:
            for _ in range(args.samples_per_video):
                try:
                    item = build_tuple(record, args, rng, len(metadata))
                except Exception as exc:
                    skipped += 1
                    pbar.write(f"[skip] {record.get('video_path', '<missing>')}: {exc}")
                    continue
                metadata.append(item)
    else:
        max_attempts = args.max_attempts or max(args.num_tuples * 20, len(records))
        pbar = tqdm(total=args.num_tuples, desc="Preparing PRD tuples")
        for attempt in range(max_attempts):
            if len(metadata) >= args.num_tuples:
                break
            record = records[attempt % len(records)]
            try:
                item = build_tuple(record, args, rng, len(metadata))
            except Exception as exc:
                skipped += 1
                if skipped <= 100:
                    pbar.write(f"[skip] {record.get('video_path', '<missing>')}: {exc}")
                continue
            metadata.append(item)
            pbar.update(1)
        pbar.close()

    metadata_path = args.output_dir / "tuples.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Saved {len(metadata)} tuples to {args.output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Skipped attempts: {skipped}")
    if args.num_tuples is not None and len(metadata) < args.num_tuples:
        print(f"WARNING: requested {args.num_tuples} tuples but generated {len(metadata)}")


if __name__ == "__main__":
    main()
