#!/usr/bin/env python3
"""Fine-tune YOLOv8n for Duckietown on this workstation and export ONNX.

This machine has no NVIDIA GPU, so training runs on the 16 CPU cores. A
nano model at 320 px on ~2-3k images takes a few hours on CPU.

Datasets, runs and the venv live outside the repo in $QUACKCOMM_TRAIN_ROOT
(default ~/quackcomm-training), because `dts devel run --sync` rsyncs the
whole project directory to the robot and ignores .dockerignore.

    T=~/quackcomm-training
    python3 training/train.py $T/out/duckietown/data.yaml            # train + export
    python3 training/train.py $T/out/duckietown/data.yaml --epochs 5 # quick check
    python3 training/train.py --export-only $T/runs/duckietown/yolov8n_320/weights/best.pt

The exported file is copied to packages/purepursuit/models/duckietown_yolov8n_320.onnx.
"""

import argparse
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_OUT = os.path.join(REPO, "packages", "purepursuit", "models", "duckietown_yolov8n_320.onnx")
TRAIN_ROOT = os.path.expanduser(os.environ.get("QUACKCOMM_TRAIN_ROOT", "~/quackcomm-training"))
RUNS = os.path.join(TRAIN_ROOT, "runs", "duckietown")


def export(weights, imgsz):
    from ultralytics import YOLO
    # FP32, static 320x320, opset 12: runs on onnxruntime CPU on the Nano and
    # can later be turned into a TensorRT engine for the Nano GPU.
    path = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=12, half=False,
                                dynamic=False, simplify=True)
    # Recent onnx/torch write IR version 10 (and torch>=2.9 forces opset 18).
    # The robot image pins onnxruntime 1.16.3, which reads IR <= 9 and opset <= 19.
    # A YOLO graph uses nothing from IR 10, so lowering the number is lossless.
    import onnx
    model = onnx.load(path)
    if model.ir_version > 9:
        model.ir_version = 9
        onnx.checker.check_model(model)
        onnx.save(model, path)
    opset = max(entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx"))
    if opset > 19:
        raise SystemExit("opset %d is too new for onnxruntime 1.16.3 on the robot" % opset)
    shutil.copyfile(path, MODEL_OUT)
    print("Exported %s -> %s" % (path, MODEL_OUT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", help="data.yaml written by merge_datasets.py")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-only", metavar="WEIGHTS")
    args = parser.parse_args()

    if args.export_only:
        export(args.export_only, args.imgsz)
        return 0
    if not args.data:
        parser.error("data.yaml is required unless --export-only is used")

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=os.path.abspath(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=min(8, os.cpu_count() or 4),
        patience=15,
        project=RUNS,
        name="yolov8n_%d" % args.imgsz,
        exist_ok=True,
        # The Duckiebot camera never sees the world upside down; colour and
        # brightness change a lot between rooms, so keep HSV augmentation.
        flipud=0.0,
        fliplr=0.5,
        hsv_v=0.5,
    )
    best = os.path.join(RUNS, "yolov8n_%d" % args.imgsz, "weights", "best.pt")
    export(best, args.imgsz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
