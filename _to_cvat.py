#!/usr/bin/env python3
"""
Pre-label faces using InsightFace's SCRFD detector, then import into CVAT.

Writes one or both of:
  - annotations.xml      "CVAT for images 1.1" format   (most reliable CVAT import)
  - yolo_labels.zip      "Ultralytics YOLO Detection" format (also importable into
                          CVAT, and the same label layout YOLOv8/11/26 train on)
  - yolo_labels/         the same YOLO files unzipped, for direct use

Setup (once):
    pip install insightface onnxruntime opencv-python tqdm
    # NVIDIA GPU? use onnxruntime-gpu instead of onnxruntime

Usage:
    python scrfd_to_cvat.py /path/to/images                    # both formats (default)
    python scrfd_to_cvat.py /path/to/images --format cvat
    python scrfd_to_cvat.py /path/to/images --format yolo -o my_output_folder

Import into CVAT (task must have a label named exactly "face", or pass --label):
    Task -> Actions -> Upload annotations, then either
      * format "CVAT 1.1"                      -> annotations.xml (zip it if CVAT asks)
      * format "Ultralytics YOLO Detection 1.0" -> yolo_labels.zip
    Then open the job and correct: delete false boxes, add missed faces.

Image names: CVAT matches annotations to images by file name. Paths are written
relative to the folder you pass in (e.g. "cam1/0001.jpg"). If you uploaded the
images to CVAT flat (no subfolders), use --basename.
"""

import argparse
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)


def load_detector(model_pack: str, det_size: int, thresh: float, cpu: bool):
    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"] if cpu else [
        "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"
    ]
    # Only load the detection model (SCRFD), skip recognition/landmark/age models.
    app = FaceAnalysis(name=model_pack, allowed_modules=["detection"], providers=providers)
    app.prepare(ctx_id=-1 if cpu else 0, det_size=(det_size, det_size), det_thresh=thresh)
    return app


def clip_box(x1, y1, x2, y2, w, h):
    return max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)


# ---------------------------------------------------------------- writers

def write_cvat_xml(results, label, out_path: Path):
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        "  <version>1.1</version>",
    ]
    for idx, r in enumerate(results):
        lines.append(f"  <image id={quoteattr(str(idx))} name={quoteattr(r['name'])} "
                     f"width={quoteattr(str(r['w']))} height={quoteattr(str(r['h']))}>")
        for x1, y1, x2, y2 in r["boxes"]:
            lines.append(
                f"    <box label={quoteattr(label)} source=\"auto\" occluded=\"0\" "
                f"xtl=\"{x1:.2f}\" ytl=\"{y1:.2f}\" xbr=\"{x2:.2f}\" ybr=\"{y2:.2f}\" z_order=\"0\"/>"
            )
        lines.append("  </image>")
    lines.append("</annotations>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yolo(results, label, out_dir: Path):
    """Ultralytics YOLO Detection layout: data.yaml, train.txt, labels/train/*.txt.
    One line per box: <class> <x_center> <y_center> <width> <height>, normalised 0-1."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    labels_root = out_dir / "labels" / "train"
    labels_root.mkdir(parents=True)

    train_list = []
    for r in results:
        rel = Path(r["name"])
        txt_path = labels_root / rel.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = r["w"], r["h"]
        rows = []
        for x1, y1, x2, y2 in r["boxes"]:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            rows.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # Empty file for images with no faces = "checked, nothing here" (useful negatives).
        txt_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        train_list.append(f"images/train/{rel.as_posix()}")

    (out_dir / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (out_dir / "data.yaml").write_text(
        "path: ./\n"
        "train: train.txt\n"
        "names:\n"
        f"  0: {label}\n",
        encoding="utf-8",
    )
    zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    return Path(zip_path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="SCRFD face pre-labels -> CVAT 1.1 XML and/or YOLO")
    ap.add_argument("images", type=Path, help="Folder of images")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("prelabels"),
                    help="Where to write the outputs (default: ./prelabels)")
    ap.add_argument("--format", choices=["both", "cvat", "yolo"], default="both",
                    help="Output format(s) (default: both)")
    ap.add_argument("--label", default="face", help='Label name (default: "face")')
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="Detection confidence threshold. Lower = more boxes, more junk to delete")
    ap.add_argument("--min-size", type=int, default=0,
                    help="Drop boxes smaller than this many pixels on the short side")
    ap.add_argument("--det-size", type=int, default=640,
                    help="Detector input size. Try 1024+ for tiny faces in big frames (slower)")
    ap.add_argument("--model", default="buffalo_l",
                    help="InsightFace model pack (buffalo_l = SCRFD-10G, buffalo_s = SCRFD-500M, faster)")
    ap.add_argument("--basename", action="store_true",
                    help="Write only file names, not relative paths")
    ap.add_argument("--no-recursive", action="store_true", help="Don't look in subfolders")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    if not args.images.is_dir():
        sys.exit(f"Not a folder: {args.images}")

    images = find_images(args.images, recursive=not args.no_recursive)
    if not images:
        sys.exit(f"No images found in {args.images}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    app = load_detector(args.model, args.det_size, args.thresh, args.cpu)

    results, skipped, total_faces = [], [], 0
    for path in tqdm(images, desc="Detecting faces"):
        img = cv2.imread(str(path))
        if img is None:
            skipped.append(path)
            continue
        h, w = img.shape[:2]
        name = path.name if args.basename else path.relative_to(args.images).as_posix()
        boxes = []
        for face in app.get(img):
            x1, y1, x2, y2 = clip_box(*map(float, face.bbox), w, h)
            if min(x2 - x1, y2 - y1) < args.min_size or x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
        total_faces += len(boxes)
        results.append({"name": name, "w": w, "h": h, "boxes": boxes})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if args.format in ("both", "cvat"):
        xml_path = args.out_dir / "annotations.xml"
        write_cvat_xml(results, args.label, xml_path)
        written.append(f"CVAT 1.1 -> {xml_path}")
    if args.format in ("both", "yolo"):
        yolo_dir = args.out_dir / "yolo_labels"
        zip_path = write_yolo(results, args.label, yolo_dir)
        written.append(f"YOLO     -> {zip_path}  (unzipped copy: {yolo_dir})")

    print(f"\nDone: {len(results)} images, {total_faces} faces")
    for line in written:
        print("  " + line)
    if skipped:
        print(f"Couldn't read {len(skipped)} file(s), e.g. {skipped[0]}")


if __name__ == "__main__":
    main()