
import os
import cv2
from inference_sdk import InferenceHTTPClient
from ambulance_dispatch import dispatch_ambulance

MODEL_ID = "traffic-accident-yolo8/1"
VIDEO_PATH = "data/videos/accident_test.mp4"

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY is not set. Set your Roboflow API key first."
    )

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

frame_index = 0
frame_interval = 5
confidence_threshold = 0.40
required_detections = 2
consecutive_detections = 0
alert_sent = False

# Simulated accident GPS coordinates for testing only.
ACCIDENT_LATITUDE = 16.5080
ACCIDENT_LONGITUDE = 80.6400

print("Starting accident response system...")
print("Video:", VIDEO_PATH)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % frame_interval == 0 and not alert_sent:
        try:
            result = client.infer(frame, model_id=MODEL_ID)
            predictions = result.get("predictions", [])

            valid_predictions = [
                p for p in predictions
                if float(p.get("confidence", 0))
                >= confidence_threshold
            ]

            if valid_predictions:
                consecutive_detections += 1
                print(
                    f"Frame {frame_index}: possible accident detected "
                    f"({consecutive_detections}/{required_detections})"
                )
            else:
                consecutive_detections = 0
                print(f"Frame {frame_index}: no qualifying detection")

            if consecutive_detections >= required_detections:
                print("\nPossible accident confirmed by repeated detections.")
                dispatch_ambulance(
                    ACCIDENT_LATITUDE,
                    ACCIDENT_LONGITUDE
                )
                alert_sent = True

        except Exception as error:
            consecutive_detections = 0
            print(f"Detection error at frame {frame_index}: {error}")

    frame_index += 1

cap.release()

print("\nVideo processing completed.")

if not alert_sent:
    print("No dispatch alert was generated during this test.")
