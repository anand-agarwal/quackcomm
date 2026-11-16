# YOLO weights

`duckietown_yolov8n_320.onnx` is the active model: YOLOv8n at 320 px trained on Duckietown data for `duckie`, `duckiebot` and `cone` (no cone labels yet). `config/yolo.yaml` selects it. `yolov5n.onnx` is the Ultralytics YOLOv5n v7.0 COCO fallback. Its identity is in `SOURCE.txt`. The image build copies this file into the container. The detector does not download weights when it starts.

The COCO label list contains `person`, `car`, `bus`, `truck`, `bicycle`, and `motorcycle`. It does not contain `duckie` or `duckiebot`. Those names are in the interest list so a custom model can use them, and a COCO model will simply never emit them.

## Custom model

Label images with the names in `duckietown.yaml`. The maintained path is `training/merge_datasets.py` + `training/train.py` (or `training/colab_train.ipynb` on a GPU); `train.py` also lowers the ONNX IR version to 9 so onnxruntime 1.16.3 on the robot can load it. On a workstation, not on the Jetson image:

```bash
pip install -r requirements-train.txt
python3 packages/purepursuit/scripts/export_custom_yolo.py train
python3 packages/purepursuit/scripts/export_custom_yolo.py export runs/detect/train/weights/best.pt
```

Export FP32 ONNX. Copy the file to `packages/purepursuit/models/`, set `class_names` in `config/yolo.yaml` to the training names (or leave it empty to use the names stored in the ONNX file), and rebuild the image.

## TensorRT

`scripts/build_tensorrt_engine.py` can build an engine on the Nano. The engine is valid only for that JetPack and GPU. The detector in this repository executes the ONNX file with onnxruntime. Building an engine does not by itself switch inference onto the GPU.
