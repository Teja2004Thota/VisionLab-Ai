# object_detection.py
# ============================================
# MAIN CONTROLLER for Object Detection Pipeline
# ============================================

from .detector_image import detect_image
from .detector_video import detect_video
from .detector_webcam import detect_webcam


def run_detection(
    input_type,
    file_path=None,
    model_path=None,
    show=True
):

    """
    Main dispatcher function.
    Calls the correct detector based on input type.
    """

    if input_type == "image":
        if not file_path:
            raise ValueError("Image path is required for image detection")
        return detect_image(
            image_path=file_path,
            model_path=model_path
        )

    elif input_type == "video":
        if not file_path:
            raise ValueError("Video path is required for video detection")
        return detect_video(
            video_path=file_path,
            model_path=model_path,
            show=show
        )

    elif input_type == "webcam":
        return detect_webcam(
            model_path=model_path,
            show=show
        )

    else:
        raise ValueError(
            "Invalid input_type. Choose from: image, video, webcam"
        )


# ===================== TEST RUN =====================
if __name__ == "__main__":

    # 🔁 Change this value to test different modes
    MODE = "image"   # image | video | webcam

    if MODE == "image":
        result = run_detection(
            input_type="image",
            file_path="sample.jpg"
        )

    elif MODE == "video":
        result = run_detection(
            input_type="video",
            file_path="sample_video.mp4"
        )

    elif MODE == "webcam":
        result = run_detection(
            input_type="webcam"
        )

    print("✅ Detection Finished")
    print(result)
