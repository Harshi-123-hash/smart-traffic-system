
import os
import cv2
from inference_sdk import InferenceHTTPClient

MODEL_ID = "traffic-accident-yolo8/1"
VIDEO_PATH = "data/videos/accident_test.mp4"
OUTPUT_PATH = "results/videos/accident_detection_test.mp4"

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY is not set. Set your Roboflow API key first."
    )

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25.0

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_index = 0
checked_frames = 0
detected_frames = 0
consecutive_detections = 0
max_consecutive_detections = 0

# Check one frame every 5 frames to reduce API usage.
FRAME_INTERVAL = 5
CONFIDENCE_THRESHOLD = 0.40
MIN_CONSECUTIVE_DETECTIONS = 2

while True:
    ret, frame = cap.read()

    if not ret:
        break

    annotated = frame.copy()

    if frame_index % FRAME_INTERVAL == 0:
        checked_frames += 1

        try:
            result = client.infer(frame, model_id=MODEL_ID)
            predictions = result.get("predictions", [])

            valid_predictions = [
                p for p in predictions
                if float(p.get("confidence", 0)) >= CONFIDENCE_THRESHOLD
            ]

            if valid_predictions:
                detected_frames += 1
                consecutive_detections += 1
                max_consecutive_detections = max(
                    max_consecutive_detections,
                    consecutive_detections
                )

                for p in valid_predictions:
                    x = int(p["x"])
                    y = int(p["y"])
                    w = int(p["width"])
                    h = int(p["height"])

                    x1 = max(0, x - w // 2)
                    y1 = max(0, y - h // 2)
                    x2 = min(width - 1, x + w // 2)
                    y2 = min(height - 1, y + h // 2)

                    confidence = float(p.get("confidence", 0))
                    class_name = str(p.get("class", "unknown"))

                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), (0, 0, 255), 2
                    )
                    cv2.putText(
                        annotated,
                        f"Model class {class_name}: {confidence:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                if consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
                    status = "POSSIBLE ACCIDENT - VERIFY"
                    color = (0, 0, 255)
                else:
                    status = "POSSIBLE ACCIDENT"
                    color = (0, 165, 255)

            else:
                consecutive_detections = 0
                status = "No qualifying detection"
                color = (0, 255, 0)

            print(
                f"Frame {frame_index}/{total_frames} | "
                f"Predictions: {len(valid_predictions)} | "
                f"Consecutive sampled detections: {consecutive_detections} | "
                f"Status: {status}"
            )

        except Exception as e:
            consecutive_detections = 0
            status = "INFERENCE ERROR"
            color = (0, 165, 255)
            print(f"Frame {frame_index}: Inference failed: {e}")

    else:
        status = "Checking video..."
        color = (255, 255, 255)

    cv2.putText(
        annotated,
        status,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    writer.write(annotated)
    frame_index += 1

cap.release()
writer.release()

print("\n--- TEST SUMMARY ---")
print(f"Frames in video: {total_frames}")
print(f"Frames sent to model: {checked_frames}")
print(f"Frames with qualifying detections: {detected_frames}")
print(f"Maximum consecutive sampled detections: {max_consecutive_detections}")
print(f"Output video: {OUTPUT_PATH}")

if max_consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
    print("Result: Repeated detections found. Review the output video.")
else:
    print("Result: No repeated detections confirmed. Review the video manually.")
