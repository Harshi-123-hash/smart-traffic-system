
from pathlib import Path
import cv2
from ultralytics import YOLO

from signal_controller import decide_green_time

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = BASE_DIR / "data" / "videos" / "ambulance.mp4"
AMBULANCE_MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
)
VEHICLE_MODEL_PATH = "yolov8n.pt"
OUTPUT_VIDEO = (
    BASE_DIR / "results" / "videos" / "smart_traffic_output.mp4"
)

VEHICLE_CLASSES = {2, 3, 5, 7}

# Keep ambulance priority active for 30 seconds of video time.
AMBULANCE_PRIORITY_SECONDS = 30


def get_traffic_level(vehicle_count):
    if vehicle_count <= 5:
        return "LOW"
    elif vehicle_count <= 12:
        return "MEDIUM"
    return "HIGH"


def main():
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_VIDEO.is_file():
        print("ERROR: Ambulance video not found:", INPUT_VIDEO)
        return

    if not AMBULANCE_MODEL_PATH.is_file():
        print("ERROR: Ambulance model not found:", AMBULANCE_MODEL_PATH)
        return

    print("Loading ambulance detection model...")
    ambulance_model = YOLO(str(AMBULANCE_MODEL_PATH))

    print("Loading vehicle detection model...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        print("ERROR: Could not open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        return

    frame_count = 0

    # None means no ambulance has been detected yet.
    priority_start_time = None

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1
            video_time = frame_count / fps

            ambulance_results = ambulance_model.predict(
                source=frame,
                conf=0.25,
                verbose=False,
            )

            vehicle_results = vehicle_model.predict(
                source=frame,
                conf=0.25,
                verbose=False,
            )

            annotated_frame = frame.copy()
            ambulance_detected_now = False
            vehicle_count = 0

            # ---------------- VEHICLE DETECTION ----------------
            for result in vehicle_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls.item())

                    if class_id not in VEHICLE_CLASSES:
                        continue

                    vehicle_count += 1
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 200, 0),
                        2,
                    )

                    cv2.putText(
                        annotated_frame,
                        f"{vehicle_model.names[class_id]} {confidence:.2f}",
                        (max(5, x1), max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # ---------------- AMBULANCE DETECTION ----------------
            for result in ambulance_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls.item())

                    if class_id != 0:
                        continue

                    ambulance_detected_now = True
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        3,
                    )

                    cv2.putText(
                        annotated_frame,
                        f"AMBULANCE {confidence:.2f}",
                        (max(5, x1), max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # ---------------- LATCH PRIORITY FOR 30 SECONDS ----------------
            # Start the timer only on the first ambulance detection.
            if ambulance_detected_now and priority_start_time is None:
                priority_start_time = video_time
                print(
                    f"Ambulance detected at {video_time:.2f}s. "
                    f"Priority active for {AMBULANCE_PRIORITY_SECONDS}s."
                )

            priority_active = (
                priority_start_time is not None
                and video_time - priority_start_time
                < AMBULANCE_PRIORITY_SECONDS
            )

            traffic_level = get_traffic_level(vehicle_count)

            decision = decide_green_time(
                traffic_level,
                ambulance_detected=priority_active,
            )

            if priority_active:
                status = "AMBULANCE PRIORITY ACTIVE"
                status_color = (0, 0, 255)  # Red
                green_seconds = AMBULANCE_PRIORITY_SECONDS
            else:
                status = "NORMAL TRAFFIC"
                status_color = (0, 255, 255)  # Yellow
                green_seconds = decision.green_seconds

            # ---------------- FIXED INFORMATION PANEL ----------------
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (650, 145),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated_frame,
                f"Traffic: {traffic_level} | Vehicles: {vehicle_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Green time: {green_seconds} seconds",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                status,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated_frame)

            if frame_count % 30 == 0:
                remaining = (
                    max(
                        0,
                        AMBULANCE_PRIORITY_SECONDS
                        - (video_time - priority_start_time),
                    )
                    if priority_active
                    else 0
                )

                print(
                    f"Frame {frame_count}/{total_frames} | "
                    f"Ambulance now: {ambulance_detected_now} | "
                    f"Priority active: {priority_active} | "
                    f"Time remaining: {remaining:.1f}s"
                )

    except KeyboardInterrupt:
        print("Processing stopped by user.")

    finally:
        cap.release()
        writer.release()

    print("Processing finished.")
    print("Output saved to:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
