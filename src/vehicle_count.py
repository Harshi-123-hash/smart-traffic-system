
import os
import cv2
import json
import tempfile
import time
from collections import Counter

from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(PROJECT_DIR, "data", "videos", "accident_test.mp4")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "results", "vehicle_count_output.mp4")
STATUS_PATH = os.path.join(PROJECT_DIR, "vehicle_status.json")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
CONFIDENCE_THRESHOLD = 0.40
PROCESS_EVERY_N_FRAMES = 5


def classify_density(count):
    if count <= 5:
        return "LOW"
    elif count <= 15:
        return "MEDIUM"
    return "HIGH"



def save_status(running, count, density, frame, message, processed=0):
    status = {
        "running": running,
        "vehicle_count": count,
        "traffic_density": density,
        "last_frame": frame,
        "processed_frames": processed,
        "message": message,
        "output_video": OUTPUT_PATH
    }

    os.makedirs(PROJECT_DIR, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=PROJECT_DIR,
            delete=False,
            suffix=".tmp"
        ) as temp_file:
            json.dump(status, temp_file, indent=2)
            temp_path = temp_file.name

        for attempt in range(5):
            try:
                os.replace(temp_path, STATUS_PATH)
                temp_path = None
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.3)

    except OSError as exc:
        print(f"Could not update vehicle status: {exc}", flush=True)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

def main():
    frame_number = 0
    latest_count = 0
    latest_density = "LOW"
    processed_frames = 0
    cap = None
    writer = None
    failed = False

    save_status(True, latest_count, latest_density, 0,
                "Loading vehicle detection model")

    try:
        if not os.path.isfile(VIDEO_PATH):
            raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

        print("Loading local RF-DETR Nano model...", flush=True)
        model = RFDETRNano()

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError("Could not create output video.")

        save_status(True, latest_count, latest_density, 0,
                    "Vehicle counting in progress")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            if frame_number % PROCESS_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(
                    rgb_frame,
                    threshold=CONFIDENCE_THRESHOLD
                )

                counts = Counter()

                for class_id in detections.class_id:
                    name = str(COCO_CLASSES[int(class_id)]).lower()
                    if name in VEHICLE_CLASSES:
                        counts[name] += 1

                latest_count = sum(counts.values())
                latest_density = classify_density(latest_count)
                processed_frames += 1

                print(
                    f"Frame {frame_number}/{total_frames} | "
                    f"Vehicles: {latest_count} | "
                    f"Density: {latest_density} | "
                    f"Classes: {dict(counts)}",
                    flush=True
                )

                save_status(
                    True, latest_count, latest_density,
                    frame_number, "Vehicle counting in progress",
                    processed_frames
                )

            cv2.putText(
                frame, f"Vehicles: {latest_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Traffic: {latest_density}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            writer.write(frame)

        print("Vehicle counting completed.", flush=True)
        print(f"Sampled frames: {processed_frames}", flush=True)
        print(f"Annotated video: {OUTPUT_PATH}", flush=True)

    except Exception as exc:
        failed = True
        save_status(
            False, latest_count, latest_density, frame_number,
            f"Vehicle counting failed: {exc}", processed_frames
        )
        raise

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        if not failed:
            save_status(
                False, latest_count, latest_density, frame_number,
                "Vehicle counting completed", processed_frames
            )


if __name__ == "__main__":
    main()
