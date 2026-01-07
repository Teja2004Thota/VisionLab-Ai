from ultralytics import YOLO
from .config import DEFAULT_MODEL

_model_cache = {}

def get_model(model_path=None):
    """
    Loads YOLO model once and reuses it.
    """
    if not model_path:
        model_path = DEFAULT_MODEL

    if model_path not in _model_cache:
        print(f"[INFO] Loading model: {model_path}")
        _model_cache[model_path] = YOLO(model_path)

    return _model_cache[model_path]
