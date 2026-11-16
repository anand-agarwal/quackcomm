"""Run the vendored YOLO graph.

Backend order: a TensorRT engine built on this device, then onnxruntime, then
OpenCV DNN. The OpenCV importer rejects the vendored FP16 YOLOv5 graph, so
onnxruntime is the backend this repository actually executes. TensorRT is
attempted only when an engine file is already present.
"""

import os

import numpy as np

from pp_core.yolo_postprocess import letterbox_params


class InferenceUnavailable(Exception):
    pass


def parse_names_metadata(text):
    """Class names from an Ultralytics ONNX 'names' entry, e.g. "{0: 'duckie', 1: 'duckiebot'}"."""
    if not text:
        return []
    import ast
    try:
        names = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=int)]
        except (TypeError, ValueError):
            return []
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _letterbox(image, network_size):
    import cv2
    height, width = image.shape[:2]
    ratio, pad_x, pad_y = letterbox_params(width, height, network_size)
    resized_w = max(1, int(round(width * ratio)))
    resized_h = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h))
    canvas = np.full((network_size, network_size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    return blob


class OnnxRuntimeBackend(object):
    name = "onnxruntime"

    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        # The Nano has 4 cores shared with dt-core, the camera and sshd. Two
        # threads without busy-wait spinning keep the rest of the robot responsive.
        options.intra_op_num_threads = int(os.environ.get("PP_ORT_THREADS", "2"))
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.network_size = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 640
        self.input_type = model_input.type
        try:
            metadata = self.session.get_modelmeta().custom_metadata_map
        except Exception:
            metadata = {}
        self.class_names = parse_names_metadata(metadata.get("names"))

    def run(self, bgr_image):
        blob = _letterbox(bgr_image, self.network_size)
        if "float16" in self.input_type:
            blob = blob.astype(np.float16)
        return self.session.run(None, {self.input_name: blob})[0]


class OpenCvBackend(object):
    name = "opencv"

    def __init__(self, model_path, network_size):
        import cv2
        self.network = cv2.dnn.readNetFromONNX(model_path)
        self.network_size = int(network_size)

    def run(self, bgr_image):
        import cv2
        blob = _letterbox(bgr_image, self.network_size)
        self.network.setInput(blob)
        return self.network.forward()


class YoloRunner(object):
    def __init__(self, model_path, engine_path=None, network_size=640):
        self.backend = None
        self.backend_name = "none"
        self.class_names = []
        self.network_size = int(network_size)
        self.error = None
        self.engine_path = engine_path
        errors = []
        # An engine can be built on the Jetson with scripts/build_tensorrt_engine.py.
        # Executing it needs CUDA bindings that are not part of this process, so a
        # present engine file must not disable the ONNX runtime.
        del engine_path
        if model_path and os.path.isfile(model_path):
            try:
                self.backend = OnnxRuntimeBackend(model_path)
                self.backend_name = self.backend.name
                self.network_size = self.backend.network_size
                self.class_names = self.backend.class_names
                return
            except Exception as exc:
                errors.append("onnxruntime: %s" % exc)
            try:
                self.backend = OpenCvBackend(model_path, network_size)
                self.backend_name = self.backend.name
                self.network_size = self.backend.network_size
                return
            except Exception as exc:
                errors.append("opencv: %s" % exc)
        else:
            errors.append("model file is missing: %s" % model_path)
        self.error = "; ".join(errors)

    @property
    def ready(self):
        return self.backend is not None

    def infer(self, bgr_image):
        if not self.ready:
            raise InferenceUnavailable(self.error or "no YOLO backend is ready")
        return self.backend.run(bgr_image)
