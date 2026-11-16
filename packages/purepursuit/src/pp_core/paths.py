import os

DEFAULT_MODEL = "yolov5n.onnx"


def package_root():
    """Directory that contains models/ when running from the source tree."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def models_dir():
    try:
        import rospkg
        return os.path.join(rospkg.RosPack().get_path("purepursuit"), "models")
    except Exception:
        return os.path.join(package_root(), "models")


def resolve_model_path(model_path=""):
    """Absolute path for ~model_path.

    Empty means the vendored COCO model. A bare file name such as
    "duckietown_yolov8n_320.onnx" is looked up in the package models/ folder,
    so yolo.yaml does not need an absolute path.
    """
    if not model_path:
        return os.path.join(models_dir(), DEFAULT_MODEL)
    if os.path.isabs(model_path) or os.path.dirname(model_path):
        return model_path
    return os.path.join(models_dir(), model_path)


def default_model_path():
    return resolve_model_path("")
