import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cv2():
    import cv2

    return cv2


def is_readable_video(path):
    cv2 = load_cv2()
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    if ok:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok = frames > 0
    cap.release()
    return ok


def convert_one(args_tuple):
    src_path, dst_path, overwrite = args_tuple
    cv2 = load_cv2()

    if dst_path.exists() and not overwrite and is_readable_video(dst_path):
        return ("skip", str(src_path), "")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return ("fail", str(src_path), "cannot open source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return ("fail", str(src_path), "invalid frame size")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return ("fail", str(src_path), "cannot open writer")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # The released processed_videos are RGB-semantic frames when decoded by
        # cv2. Convert them to BGR before writing a standard OpenCV-readable mp4.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    if frame_count == 0:
        tmp_path.unlink(missing_ok=True)
        return ("fail", str(src_path), "no frames written")

    tmp_path.replace(dst_path)
    return ("ok", str(src_path), "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PRD processed_videos to standard RGB-color mp4 files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=PROJECT_ROOT / "data/PRD/train/processed_videos",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data/PRD/train/processed_videos_rgb",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "outputs/processed_videos_rgb_failures.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    videos = sorted(path for path in args.input_dir.iterdir() if path.is_file())
    if args.limit is not None:
        videos = videos[: args.limit]

    tasks = [(path, args.output_dir / path.name, args.overwrite) for path in videos]
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(convert_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting videos"):
            status, src, message = future.result()
            counts[status] += 1
            if status == "fail":
                failures.append(f"{src}\t{message}")

    if failures:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(failures) + "\n", encoding="utf-8")

    print(f"ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    if failures:
        print(f"Failure log: {args.log}")


if __name__ == "__main__":
    main()
