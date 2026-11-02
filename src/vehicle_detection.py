
from pathlib import Path
import cv2
import math
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = BASE_DIR / "data" / "videos" / "traffic.mp4"
OUTPUT_VIDEO = BASE_DIR / "results" / "videos" / "detected_traffic.mp4"

MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35

OBSERVATION_SECONDS = 3
YELLOW_SECONDS = 5

GREEN_TIME_MAPPING = {
    "LOW": 10,
    "MEDIUM": 20,
    "HIGH": 30
}

LOW_MAX = 5
MEDIUM_MAX = 15

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# --------------------------------------------------
# VEHICLE COUNTING
# --------------------------------------------------

def count_vehicles(result):

    if result.boxes is None:
        return 0

    vehicle_count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id in VEHICLE_CLASSES:
            vehicle_count += 1

    return vehicle_count


# --------------------------------------------------
# TRAFFIC DENSITY CLASSIFICATION
# --------------------------------------------------

def classify_traffic(max_vehicle_count):

    if max_vehicle_count <= LOW_MAX:
        return "LOW"

    if max_vehicle_count <= MEDIUM_MAX:
        return "MEDIUM"

    return "HIGH"


# --------------------------------------------------
# GREEN TIMER ALLOCATION
# --------------------------------------------------

def allocate_green_time(density_level):

    return GREEN_TIME_MAPPING[density_level]


# --------------------------------------------------
# TEXT DISPLAY
# --------------------------------------------------

def draw_text(frame, text, position,
              color=(255, 255, 255),
              scale=0.7,
              thickness=2):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_transparent_panel(frame, x, y, width, height,
                           color=(15, 15, 15), alpha=0.78):

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, min(frame_width, x))
    y1 = max(0, min(frame_height, y))
    x2 = max(0, min(frame_width, x + width))
    y2 = max(0, min(frame_height, y + height))

    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# --------------------------------------------------
# SIDE TRAFFIC LIGHT
# --------------------------------------------------

def draw_traffic_signal(frame, signal_state):

    height, width = frame.shape[:2]

    box_width = max(62, int(width * 0.065))
    box_height = max(165, int(height * 0.30))

    x = width - box_width - max(15, int(width * 0.025))
    y = max(18, int(height * 0.04))

    center_x = x + box_width // 2

    radius = max(15, int(box_width * 0.28))
    gap = (box_height - 2 * radius) // 2

    positions = {
        "RED": (center_x, y + radius + 5),
        "YELLOW": (center_x, y + radius + gap),
        "GREEN": (center_x, y + box_height - radius - 5)
    }

    colors = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0)
    }

    draw_transparent_panel(
        frame,
        x,
        y,
        box_width,
        box_height,
        (10, 10, 10),
        0.95
    )

    for state, center in positions.items():

        light_color = (
            colors[state]
            if state == signal_state
            else (55, 55, 55)
        )

        cv2.circle(
            frame,
            center,
            radius,
            light_color,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            center,
            radius,
            (100, 100, 100),
            1,
            cv2.LINE_AA
        )


# --------------------------------------------------
# VIDEO OVERLAY
# --------------------------------------------------

def draw_traffic_overlay(
    frame,
    signal_state,
    phase_remaining,
    current_vehicle_count,
    max_vehicle_count,
    density_level,
    green_duration,
    observation_count,
    observation_frames
):

    height, width = frame.shape[:2]

    scale = max(0.65, min(1.15, width / 1050))

    white = (255, 255, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    orange = (0, 165, 255)
    grey = (100, 100, 100)

    signal_color = {
        "RED": red,
        "YELLOW": yellow,
        "GREEN": green
    }[signal_state]

    # ----------------------------------------------
    # TOP-LEFT GREY INFORMATION PANEL
    # ----------------------------------------------

    info_x = int(12 * scale)
    info_y = int(28 * scale)

    info_width = min(
        int(440 * scale),
        int(width * 0.48)
    )

    info_height = int(125 * scale)

    draw_transparent_panel(
        frame,
        info_x - 5,
        info_y - int(23 * scale),
        info_width,
        info_height,
        grey,
        0.82
    )

    draw_text(
        frame,
        f"LIVE VEHICLES: {current_vehicle_count}",
        (info_x, info_y),
        white,
        0.68 * scale,
        2
    )

    draw_text(
        frame,
        f"MAX VEHICLES: {max_vehicle_count}",
        (info_x, info_y + int(32 * scale)),
        white,
        0.68 * scale,
        2
    )

    draw_text(
        frame,
        f"TRAFFIC LEVEL: {density_level}",
        (info_x, info_y + int(64 * scale)),
        orange,
        0.72 * scale,
        2
    )

    draw_text(
        frame,
        f"GREEN ALLOCATED: {green_duration} seconds",
        (info_x, info_y + int(96 * scale)),
        white,
        0.62 * scale,
        2
    )

    # ----------------------------------------------
    # TRAFFIC LEVEL BANNER
    # ----------------------------------------------

    banner_x = int(15 * scale)
    banner_y = int(155 * scale)

    banner_width = min(
        int(430 * scale),
        int(width * 0.60)
    )

    banner_height = int(48 * scale)

    draw_transparent_panel(
        frame,
        banner_x,
        banner_y,
        banner_width,
        banner_height,
        orange,
        0.92
    )

    draw_text(
        frame,
        f"TRAFFIC LEVEL: {density_level}",
        (
            banner_x + int(12 * scale),
            banner_y + int(33 * scale)
        ),
        white,
        0.72 * scale,
        2
    )

    # ----------------------------------------------
    # HIGHLIGHTED COUNTDOWN - UPPER MIDDLE
    # ----------------------------------------------

    timer_width = min(
        int(370 * scale),
        int(width * 0.30)
    )

    timer_height = int(130 * scale)

    timer_x = (width - timer_width) // 2
    timer_y = int(15 * scale)

    draw_transparent_panel(
        frame,
        timer_x,
        timer_y,
        timer_width,
        timer_height,
        (8, 8, 8),
        0.90
    )

    cv2.rectangle(
        frame,
        (timer_x, timer_y),
        (timer_x + timer_width, timer_y + timer_height),
        signal_color,
        max(2, int(3 * scale))
    )

    if signal_state == "GREEN":
        timer_title = "GREEN TIME REMAINING"

    elif signal_state == "YELLOW":
        timer_title = "YELLOW TIME REMAINING"

    else:
        timer_title = "RED / TRAFFIC OBSERVATION"

    draw_text(
        frame,
        timer_title,
        (
            timer_x + int(18 * scale),
            timer_y + int(34 * scale)
        ),
        signal_color,
        0.52 * scale,
        2
    )

    draw_text(
        frame,
        f"{phase_remaining:02d}",
        (
            timer_x + int(80 * scale),
            timer_y + int(105 * scale)
        ),
        signal_color,
        2.0 * scale,
        3
    )

    draw_text(
        frame,
        "SECONDS",
        (
            timer_x + int(195 * scale),
            timer_y + int(100 * scale)
        ),
        white,
        0.58 * scale,
        2
    )

    if signal_state in ("YELLOW", "RED"):

        draw_text(
            frame,
            f"OBSERVATION: {observation_count}/{observation_frames} FRAMES",
            (
                int(15 * scale),
                min(height - 12, int(385 * scale))
            ),
            white,
            0.48 * scale,
            1
        )

    # ----------------------------------------------
    # TRAFFIC LIGHT - RIGHT SIDE
    # ----------------------------------------------

    draw_traffic_signal(
        frame,
        signal_state
    )


# --------------------------------------------------
# MAIN VIDEO PROCESSING
# --------------------------------------------------

def main():

    print("\nSMART TRAFFIC MANAGEMENT SYSTEM")
    print("--------------------------------")

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Unable to open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    observation_frames = max(
        1,
        math.ceil(OBSERVATION_SECONDS * fps)
    )

    yellow_frames = max(
        1,
        math.ceil(YELLOW_SECONDS * fps)
    )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError("Unable to create output video.")

    # ----------------------------------------------
    # INITIAL CYCLE STATE
    # ----------------------------------------------

    phase = "YELLOW"
    phase_frame = 0
    frame_number = 0

    observation_count = 0
    frame_counts = []
    observation_complete = False

    current_vehicle_count = 0
    max_vehicle_count = 0
    density_level = "LOW"
    green_duration = 10

    print(f"Input video: {INPUT_VIDEO}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Observation duration: {OBSERVATION_SECONDS} seconds")
    print(f"Yellow duration: {YELLOW_SECONDS} seconds")
    print("Signal sequence: YELLOW -> GREEN -> RED -> YELLOW")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ------------------------------------------
            # YOLO DETECTION ON EVERY FRAME
            # ------------------------------------------

            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            result = results[0]

            current_vehicle_count = count_vehicles(result)

            annotated_frame = result.plot()

            # ------------------------------------------
            # PHASE 1: YELLOW - 5 SECONDS
            # ------------------------------------------

            if phase == "YELLOW":

                # Observe traffic during the first 3 seconds.

                if not observation_complete:

                    frame_counts.append(current_vehicle_count)
                    observation_count += 1

                    if observation_count >= observation_frames:

                        max_vehicle_count = max(
                            frame_counts,
                            default=0
                        )

                        density_level = classify_traffic(
                            max_vehicle_count
                        )

                        green_duration = allocate_green_time(
                            density_level
                        )

                        observation_complete = True

                        print("\nTRAFFIC OBSERVATION COMPLETED")
                        print("--------------------------------")
                        print(f"Observed frame counts: {frame_counts}")
                        print(f"Maximum vehicle count: {max_vehicle_count}")
                        print(f"Selected density: {density_level}")
                        print(f"Allocated green time: {green_duration} seconds")

                phase_remaining = max(
                    0,
                    math.ceil((yellow_frames - phase_frame) / fps)
                )

                phase_frame += 1

                if (
                    phase_frame >= yellow_frames
                    and observation_complete
                ):

                    phase = "GREEN"
                    phase_frame = 0

            # ------------------------------------------
            # PHASE 2: GREEN - TRAFFIC-BASED TIMER
            # ------------------------------------------

            elif phase == "GREEN":

                elapsed_seconds = phase_frame / fps

                phase_remaining = max(
                    0,
                    green_duration - int(elapsed_seconds)
                )

                green_frames = max(
                    1,
                    math.ceil(green_duration * fps)
                )

                phase_frame += 1

                if phase_frame >= green_frames:

                    # Green completed. Switch to red.

                    phase = "RED"
                    phase_frame = 0

                    observation_count = 0
                    frame_counts = []
                    max_vehicle_count = 0
                    observation_complete = False

                    print("\nGREEN SIGNAL COMPLETED.")
                    print("Switching to RED for the next observation.")

            # ------------------------------------------
            # PHASE 3: RED - NEXT CYCLE OBSERVATION
            # ------------------------------------------

            elif phase == "RED":

                frame_counts.append(current_vehicle_count)
                observation_count += 1

                phase_remaining = max(
                    0,
                    math.ceil((observation_frames - phase_frame) / fps)
                )

                phase_frame += 1

                if phase_frame >= observation_frames:

                    max_vehicle_count = max(
                        frame_counts,
                        default=0
                    )

                    density_level = classify_traffic(
                        max_vehicle_count
                    )

                    green_duration = allocate_green_time(
                        density_level
                    )

                    print("\nTRAFFIC OBSERVATION COMPLETED")
                    print("--------------------------------")
                    print(f"Observed frame counts: {frame_counts}")
                    print(f"Maximum vehicle count: {max_vehicle_count}")
                    print(f"Selected density: {density_level}")
                    print(f"Allocated green time: {green_duration} seconds")

                    # Start another 5-second yellow phase.

                    phase = "YELLOW"
                    phase_frame = 0
                    observation_count = 0
                    frame_counts = []
                    observation_complete = True

            # ------------------------------------------
            # DRAW OVERLAYS ON ORIGINAL VIDEO
            # ------------------------------------------

            draw_traffic_overlay(
                annotated_frame,
                phase,
                phase_remaining,
                current_vehicle_count,
                max_vehicle_count,
                density_level,
                green_duration,
                observation_count,
                observation_frames
            )

            out.write(annotated_frame)

            cv2.imshow(
                "Smart Traffic Management System",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print("\nVIDEO PROCESSING COMPLETED")
    print("--------------------------------")
    print(f"Processed frames: {frame_number}")
    print(f"Output saved at: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()