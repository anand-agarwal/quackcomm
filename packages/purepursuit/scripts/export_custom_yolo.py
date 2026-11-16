#!/usr/bin/env python3
"""Train and export a Duckietown ONNX model on a workstation.

Do not run this inside the Duckiebot image and do not add ultralytics to
dependencies-py3.txt. The robot image only needs the exported ONNX file.

Example, from a CUDA workstation:

    pip install -r requirements-train.txt
    python3 packages/purepursuit/scripts/export_custom_yolo.py train
    python3 packages/purepursuit/scripts/export_custom_yolo.py export runs/detect/train/weights/best.pt
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "export"):
        print("Usage: export_custom_yolo.py train|export [weights.pt]")
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install requirements-train.txt on a workstation.")
        return 1
    if sys.argv[1] == "train":
        model = YOLO("yolov5nu.pt")
        model.train(data="packages/purepursuit/models/duckietown.yaml", imgsz=320, epochs=50, batch=16)
        return 0
    weights = sys.argv[2] if len(sys.argv) > 2 else "yolov5nu.pt"
    model = YOLO(weights)
    exported = model.export(format="onnx", imgsz=320, opset=12, half=False, simplify=True)
    print(exported)
    print("Copy the ONNX file to packages/purepursuit/models/ and set yolo.yaml class_names.")
    print("Use FP32. The OpenCV DNN importer on several Jetson images rejects FP16.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
