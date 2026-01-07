import os

# Project root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Upload paths
UPLOAD_IMAGE_DIR = os.path.join(OUTPUT_DIR, "uploads", "image")
UPLOAD_VIDEO_DIR = os.path.join(OUTPUT_DIR, "uploads", "video")

# Detection output paths
DETECTED_IMAGE_DIR = os.path.join(OUTPUT_DIR, "detected", "image")
DETECTED_VIDEO_DIR = os.path.join(OUTPUT_DIR, "detected", "video")

# Reports
IMAGE_REPORT_DIR = os.path.join(DETECTED_IMAGE_DIR, "reports")
VIDEO_REPORT_DIR = os.path.join(DETECTED_VIDEO_DIR, "reports")

# Create all folders
for path in [
    UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR,
    DETECTED_IMAGE_DIR, DETECTED_VIDEO_DIR,
    IMAGE_REPORT_DIR, VIDEO_REPORT_DIR
]:
    os.makedirs(path, exist_ok=True)

# Webcam detected outputs
DETECTED_WEBCAM_DIR = os.path.join(OUTPUT_DIR, "detected", "webcam")
WEBCAM_REPORT_DIR = os.path.join(DETECTED_WEBCAM_DIR, "reports")

# Create webcam folders
for path in [DETECTED_WEBCAM_DIR, WEBCAM_REPORT_DIR]:
    os.makedirs(path, exist_ok=True)
