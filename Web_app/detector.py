from ultralytics import YOLO
import cv2

def run_detection(model_name, input_type, file_path):

    # Select model
    if model_name == "yolov8":
        model = YOLO("yolov8x.pt")
    else:
        model = YOLO("yolov5s.pt")

    # IMAGE
    if input_type == "image":
        img = cv2.imread(file_path)
        results = model(img)
        results[0].save(filename="outputs/result.jpg")

    # VIDEO
    elif input_type == "video":
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    # WEBCAM
    elif input_type == "webcam":
        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            results = model(frame)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
