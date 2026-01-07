from flask import Flask, render_template, request, send_from_directory
import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from object_detection.object_detection import run_detection
from object_detection.paths import (
    UPLOAD_IMAGE_DIR,
    UPLOAD_VIDEO_DIR,
    DETECTED_IMAGE_DIR,
    DETECTED_VIDEO_DIR,
    DETECTED_WEBCAM_DIR
)

app = Flask(__name__)


# ===================== STATIC FILE ROUTES =====================

@app.route("/uploads/image/<path:filename>")
def uploaded_image(filename):
    return send_from_directory(UPLOAD_IMAGE_DIR, filename)


@app.route("/uploads/video/<path:filename>")
def uploaded_video(filename):
    return send_from_directory(UPLOAD_VIDEO_DIR, filename)


@app.route("/detected/image/<path:filename>")
def detected_image(filename):
    return send_from_directory(DETECTED_IMAGE_DIR, filename)


@app.route("/detected/video/<path:filename>")
def detected_video(filename):
    return send_from_directory(DETECTED_VIDEO_DIR, filename)

@app.route("/detected/webcam/<path:filename>")
def detected_webcam(filename):
    return send_from_directory(DETECTED_WEBCAM_DIR, filename)

# ===================== MAIN ROUTE =====================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    video_url = None
    uploaded_image_url = None
    webcam_message = None

    if request.method == "POST":
        input_type = request.form.get("input_type")
        model_path = request.form.get("model_path")

        # ---------- IMAGE ----------
        if input_type == "image":
            file = request.files.get("file")
            if file:
                file_path = os.path.join(UPLOAD_IMAGE_DIR, file.filename)
                file.save(file_path)

                uploaded_image_url = f"/uploads/image/{file.filename}"

                result = run_detection(
                    input_type="image",
                    file_path=file_path,
                    model_path=model_path,
                    show=False
                )

                image_filename = os.path.basename(result["output_image"])
                image_url = f"/detected/image/{image_filename}"

        # ---------- VIDEO ----------
        elif input_type == "video":
            file = request.files.get("file")
            if file:
                file_path = os.path.join(UPLOAD_VIDEO_DIR, file.filename)
                file.save(file_path)

                result = run_detection(
                    input_type="video",
                    file_path=file_path,
                    model_path=model_path,
                    show=True
                )

                video_filename = os.path.basename(result["output_video"])
                video_url = f"/detected/video/{video_filename}"

        # ---------- WEBCAM ----------
        elif input_type == "webcam":
            result = run_detection(
                input_type="webcam",
                model_path=model_path,
                show=True
            )

            webcam_message = (
                "Webcam detection started using selected YOLOv8 model. "
                "Check OpenCV window. Press Q to stop."
            )

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        video_url=video_url,
        uploaded_image_url=uploaded_image_url,
        webcam_message=webcam_message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

