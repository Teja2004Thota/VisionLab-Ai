# detector_image.py
# ============================================
# IMAGE-ONLY Object Detection using YOLOv8
# ============================================

import cv2
import time
import os
import pandas as pd
from datetime import datetime

from .model_manager import get_model
from .paths import DETECTED_IMAGE_DIR, IMAGE_REPORT_DIR

# ===================== CONFIG =====================
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ===================== MAIN FUNCTION =====================
def detect_image(image_path, model_path=None):
    """
    Performs object detection on a single image.
    Returns detection summary dictionary for frontend display.
    """

    # ---------- Load Model ----------
    model = get_model(model_path)

    # ---------- Load Image ----------
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    detections = []
    class_counts = {}
    class_conf_sum = {}

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- Inference ----------
    results = model(frame)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detections.append(confidence)

            # Stats
            class_counts[label] = class_counts.get(label, 0) + 1
            class_conf_sum[label] = class_conf_sum.get(label, 0) + confidence

    # ---------- Performance ----------
    inference_time = time.time() - start_time
    fps = 1 / inference_time if inference_time > 0 else 0
    avg_conf = sum(detections) / len(detections) if detections else 0

    class_avg_conf = {
        cls: class_conf_sum[cls] / class_counts[cls]
        for cls in class_counts
    }

    # ---------- Save Image ----------
    output_image_path = os.path.join(
        DETECTED_IMAGE_DIR, "detected_image.jpg"
    )
    cv2.imwrite(output_image_path, frame)

    # ---------- Save TXT Report ----------
    report_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(
        IMAGE_REPORT_DIR, f"image_report_{report_timestamp}.txt"
    )

    with open(report_path, "w") as f:
        f.write("MODE: IMAGE\n")
        f.write(f"Total Detections: {len(detections)}\n")
        f.write(f"Average Confidence: {avg_conf:.2f}\n")
        f.write(f"Inference Time (ms): {inference_time * 1000:.2f}\n")
        f.write(f"FPS (Derived): {fps:.2f}\n\n")

        f.write("Per-Class Statistics:\n")
        for cls in class_counts:
            f.write(
                f"{cls}: Count={class_counts[cls]}, "
                f"AvgConf={class_avg_conf[cls]:.2f}\n"
            )

    # ---------- Return Frontend-Friendly Summary ----------
    return {
        "mode": "image",
        "output_image": output_image_path,
        "report": report_path,
        "detections": len(detections),
        "fps": fps,
        "inference_time_ms": inference_time * 1000,
        "avg_confidence": avg_conf,
        "class_counts": class_counts,
        "class_avg_confidence": class_avg_conf
    }


# ===================== TEST RUN =====================
if __name__ == "__main__":
    IMAGE_PATH = "sample.jpg"
    summary = detect_image(IMAGE_PATH)
    print("✅ Image Detection Completed")
    print(summary)
