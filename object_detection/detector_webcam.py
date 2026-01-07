# detector_webcam.py
# ============================================
# WEBCAM-ONLY Object Detection using YOLOv8
# ============================================

import cv2
import time
import os
from datetime import datetime
from collections import defaultdict

from .model_manager import get_model
from .paths import DETECTED_WEBCAM_DIR, WEBCAM_REPORT_DIR

# ===================== CONFIG =====================
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
CAMERA_INDEX =1   # default webcam (change if needed)



# ===================== MAIN FUNCTION =====================
def detect_webcam(model_path=None, show=True):
    """
    Performs real-time object detection using webcam.
    Shows live feed and saves detected webcam video.
    Press 'Q' to stop.
    Returns frontend-ready summary dictionary.
    """

    # ---------- Load Model ----------
    model = get_model(model_path)

    # ---------- Open Webcam ----------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Unable to open webcam at index {CAMERA_INDEX}"
    )


    # ---------- Output Webcam Video ----------
    output_video_path = os.path.join(
        DETECTED_WEBCAM_DIR, "detected_webcam.mp4"
    )

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        25,  # webcam FPS fallback
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    if not out.isOpened():
        raise RuntimeError("❌ Failed to open VideoWriter for webcam")

    frame_count = 0
    confidences = []

    class_counts = defaultdict(int)
    class_conf_sum = defaultdict(float)

    start_time = time.time()

    # ---------- Webcam Loop ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

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
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
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

        # ---------- FPS ----------
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(frame)

        if show:
            cv2.imshow("YOLOv8 Webcam Detection", frame)

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
        WEBCAM_REPORT_DIR, f"webcam_report_{report_timestamp}.txt"
    )

    with open(report_path, "w") as f:
        f.write("MODE: WEBCAM\n")
        f.write(f"Camera Index: {CAMERA_INDEX}\n")
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
        "mode": "webcam",
        "frames": frame_count,
        "detections": len(confidences),
        "fps": fps,
        "avg_confidence": avg_confidence,
        "class_counts": dict(class_counts),
        "class_avg_confidence": class_avg_confidence,
        "output_video": output_video_path,
        "report": report_path
    }


# ===================== TEST RUN =====================
if __name__ == "__main__":
    summary = detect_webcam(show=True)
    print("✅ Webcam Detection Completed")
    print(summary)
