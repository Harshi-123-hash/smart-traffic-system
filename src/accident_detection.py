
from pathlib import Path
import os
import cv2
from datetime import datetime
from inference_sdk import InferenceHTTPClient

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "data" / "videos" / "accident_test.mp4"

OUTPUT_PATH = (
    BASE_DIR / "results" / "videos" / "accident_detected_output.mp4"
)

# Save photos of frames where an accident prediction occurs.
PHOTOS_DIR = BASE_DIR / "results" / "accident_frames"

MODEL_ID = "traffic-accident-yolo8/1"
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# =====================================================
# 2. SETTINGS
# =====================================================

CONFIDENCE_THRESHOLD = 0.40
INFERENCE_EVERY_N_FRAMES = 5

# OpenCV uses BGR color format.
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# =====================================================
# 3. DRAW RED ACCIDENT STATUS
# =====================================================

def draw_accident_status(frame):
    text = "ACCIDENT DETECTED"

    x = 30
    y = 55

    # Black outline improves visibility.
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, BLACK, 6, cv2.LINE_AA
    )

    # Red status text.
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, RED, 3, cv2.LINE_AA
    )

    return frame


# =====================================================
# 4. DRAW RED BOXES AROUND DETECTED ACCIDENT AREAS
# =====================================================

def draw_predictions(frame, predictions):
    height, width = frame.shape[:2]

    for prediction in predictions:
        confidence = float(prediction.get("confidence", 0))

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Roboflow provides center coordinates and box dimensions.
        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        box_width = float(prediction.get("width", 0))
        box_height = float(prediction.get("height", 0))

        x1 = max(0, int(x - box_width / 2))
        y1 = max(0, int(y - box_height / 2))
        x2 = min(width - 1, int(x + box_width / 2))
        y2 = min(height - 1, int(y + box_height / 2))

        # Red rectangle around the model's predicted area.
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), RED, 4
        )

        label = (
            f"{prediction.get('class', 'accident')} "
            f"{confidence:.2f}"
        )

        label_y = max(25, y1 - 10)

        cv2.putText(
            frame, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, RED, 2, cv2.LINE_AA
        )

    return frame


# =====================================================
# 5. SAVE ANNOTATED ACCIDENT PHOTOS
# =====================================================

def save_accident_photo(frame, frame_number, fps):
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = frame_number / fps
    minutes = int(timestamp // 60)
    seconds = timestamp % 60

    filename = (
        f"accident_frame_{frame_number:06d}_"
        f"{minutes:02d}m{seconds:05.2f}s.jpg"
    )

    photo_path = PHOTOS_DIR / filename

    if cv2.imwrite(str(photo_path), frame):
        print(f"Accident photo saved: {photo_path}")
    else:
        print(f"WARNING: Could not save photo: {photo_path}")


# =====================================================
# 6. MAIN VIDEO PROCESSING
# =====================================================

def main():
    if not API_KEY:
        print("ERROR: Roboflow API key is missing.")
        print("Configure ROBOFLOW_API_KEY in this terminal.")
        return

    if not VIDEO_PATH.is_file():
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        return

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

    frame_number = 0
    accident_detected = False

    # Keep the last accident boxes for the remainder of the video.
    persistent_accident_boxes = []

    # Avoid saving duplicate photos for the same sampled frame.
    saved_frames = set()

    print("Starting accident detection...")
    print("Input video:", VIDEO_PATH)
    print("Output video:", OUTPUT_PATH)
    print("Accident photos folder:", PHOTOS_DIR)
    print("Total frames:", total_frames)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            should_infer = (
                (frame_number - 1) % INFERENCE_EVERY_N_FRAMES == 0
            )

            if should_infer:
                try:
                    response = client.infer(
                        frame,
                        model_id=MODEL_ID
                    )

                    predictions = response.get("predictions", [])

                    qualifying_predictions = [
                        p for p in predictions
                        if float(p.get("confidence", 0))
                        >= CONFIDENCE_THRESHOLD
                    ]

                    if qualifying_predictions:
                        if not accident_detected:
                            accident_detected = True
                            print(
                                f"ACCIDENT DETECTED at frame "
                                f"{frame_number} "
                                f"(video time: {frame_number / fps:.2f}s)"
                            )

                        # Update the remembered accident location.
                        persistent_accident_boxes = qualifying_predictions

                        # Save the frame with its red box and red status.
                        photo_frame = frame.copy()
                        photo_frame = draw_predictions(
                            photo_frame,
                            qualifying_predictions
                        )
                        photo_frame = draw_accident_status(photo_frame)

                        if frame_number not in saved_frames:
                            save_accident_photo(
                                photo_frame,
                                frame_number,
                                fps
                            )
                            saved_frames.add(frame_number)

                except Exception as error:
                    print(
                        f"Inference error at frame {frame_number}: "
                        f"{error}"
                    )

            # Keep the last detected accident box visible.
            # This is most appropriate for a fixed CCTV camera.
            if accident_detected and persistent_accident_boxes:
                frame = draw_predictions(
                    frame,
                    persistent_accident_boxes
                )

            # Never display "ACCIDENT NOT DETECTED".
            # Only display the red status after detection occurs.
            if accident_detected:
                frame = draw_accident_status(frame)

            writer.write(frame)

            if frame_number % 100 == 0:
                print(
                    f"Progress: {frame_number}/{total_frames}"
                )

    finally:
        cap.release()
        writer.release()

    print("\nProcessing completed.")
    print("Frames processed:", frame_number)
    print("Accident detected:", accident_detected)
    print("Annotated video:", OUTPUT_PATH)
    print("Saved accident photos:", PHOTOS_DIR)

    if not accident_detected:
        print(
            "No qualifying accident predictions were found. "
            "No accident photos were saved."
        )


if __name__ == "__main__":
    main()
