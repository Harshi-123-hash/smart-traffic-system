
from pathlib import Path
import os
import cv2
from inference_sdk import InferenceHTTPClient

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "videos" / "accident_test.mp4"

MODEL_ID = "accident-detection-model/2"
API_KEY = os.getenv("ROBOFLOW_API_KEY")


def main():
    if not API_KEY:
        print("ERROR: ROBOFLOW_API_KEY is missing.")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open video:", VIDEO_PATH)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total)

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

    # Test five frames around the middle of the video.
    middle = total // 2
    indices = sorted(set(
        max(0, min(total - 1, middle + offset))
        for offset in [-20, -10, 0, 10, 20]
    ))

    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()

            if not success:
                print("Could not read frame:", index)
                continue

            print(f"\n--- Frame {index} ---")

            try:
                response = client.infer(frame, model_id=MODEL_ID)
                print("Full response:", response)
                predictions = response.get("predictions", [])
                print("Prediction count:", len(predictions))
            except Exception as error:
                print("Inference error:", repr(error))

    finally:
        cap.release()


if __name__ == "__main__":
    main()
