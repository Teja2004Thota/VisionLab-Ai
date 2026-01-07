# detector_video.py
# ============================================
# VIDEO-ONLY Object Detection using YOLOv8
# ============================================

import cv2
import time
import os
from datetime import datetime
from collections import defaultdict

from .model_manager import get_model
from .paths import DETECTED_VIDEO_DIR, VIDEO_REPORT_DIR

# ===================== CONFIG =====================
CONF_THRESHOLD = 0.5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


# ===================== MAIN FUNCTION =====================
def detect_video(video_path, model_path=None, show=False):
    """
    Performs object detection on a video file.
    Saves detected video and generates performance metrics.
    Returns frontend-ready summary dictionary.
    """

    # ---------- Load Model ----------
    model = get_model(model_path)

    # ---------- Open Video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- Output Video ----------
    output_video_path = os.path.join(
        DETECTED_VIDEO_DIR, "detected_video.mp4"
    )

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps_input if fps_input > 0 else 25,
        (width, height)
    )

    frame_count = 0
    confidences = []

    class_counts = defaultdict(int)
    class_conf_sum = defaultdict(float)

    start_time = time.time()

    # ---------- Frame Loop ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(frame, stream=True)

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

                # Stats
                confidences.append(confidence)
                class_counts[label] += 1
                class_conf_sum[label] += confidence

        out.write(frame)

        if show:
            cv2.imshow("YOLOv8 Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ---------- Cleanup ----------
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ---------- Performance ----------
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    class_avg_confidence = {
        cls: class_conf_sum[cls] / class_counts[cls]
        for cls in class_counts
    }

    # ---------- Save TXT Report ----------
    report_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(
        VIDEO_REPORT_DIR, f"video_report_{report_timestamp}.txt"
    )

    with open(report_path, "w") as f:
        f.write("MODE: VIDEO\n")
        f.write(f"Total Frames: {frame_count}\n")
        f.write(f"Total Detections: {len(confidences)}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Average Confidence: {avg_confidence:.2f}\n\n")

        f.write("Per-Class Statistics:\n")
        for cls in class_counts:
            f.write(
                f"{cls}: Count={class_counts[cls]}, "
                f"AvgConf={class_avg_confidence[cls]:.2f}\n"
            )

    # ---------- Return Frontend Summary ----------
    return {
        "mode": "video",
        "output_video": output_video_path,
        "frames": frame_count,
        "detections": len(confidences),
        "fps": fps,
        "avg_confidence": avg_confidence,
        "class_counts": dict(class_counts),
        "class_avg_confidence": class_avg_confidence,
        "report": report_path
    }


# ===================== TEST RUN =====================
if __name__ == "__main__":
    VIDEO_PATH = "sample_video.mp4"
    summary = detect_video(VIDEO_PATH, show=True)
    print("✅ Video Detection Completed")
    print(summary)
