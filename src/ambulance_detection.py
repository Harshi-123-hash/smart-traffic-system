
from pathlib import Path
import cv2
import math
from ultralytics import YOLO


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = BASE_DIR / "data" / "videos" / "ambulance.mp4"

AMBULANCE_MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
)

VEHICLE_MODEL_PATH = "yolov8n.pt"

OUTPUT_VIDEO = (
    BASE_DIR / "results" / "videos" / "smart_traffic_output.mp4"
)


# ==========================================================
# 2. SYSTEM SETTINGS
# ==========================================================

VEHICLE_CLASSES = {2, 3, 5, 7}

RED_OBSERVATION_SECONDS = 30
YELLOW_SECONDS = 5
AMBULANCE_PRIORITY_SECONDS = 20

AMBULANCE_CONFIDENCE = 0.15
AMBULANCE_IMAGE_SIZE = 960

VEHICLE_CONFIDENCE = 0.25

BOX_PERSISTENCE_SECONDS = 0.5

# Traffic classification thresholds
LOW_MAX = 5
MEDIUM_MAX = 15

# Green-time allocation
GREEN_TIME_MAPPING = {
    "LOW": 10,
    "MEDIUM": 20,
    "HIGH": 30
}


# ==========================================================
# 3. COLORS (BGR FORMAT)
# ==========================================================

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREY = (90, 90, 90)
ORANGE = (0, 165, 255)


# ==========================================================
# 4. AMBULANCE MODEL CLASS IDENTIFICATION
# ==========================================================

def get_ambulance_class_ids(model):

    names = model.names

    if isinstance(names, list):
        names = {
            index: name
            for index, name in enumerate(names)
        }

    ambulance_ids = set()

    for class_id, class_name in names.items():

        normalized_name = str(class_name).lower().strip()

        if (
            "ambulance" in normalized_name
            or "storbow" in normalized_name
            or "emergency vehicle" in normalized_name
        ):
            ambulance_ids.add(int(class_id))

    # Support the previously trained single-class model.
    if not ambulance_ids and len(names) == 1:
        ambulance_ids.add(int(next(iter(names.keys()))))

    print("\nAMBULANCE MODEL CLASSES")
    print("--------------------------------")
    print("Model classes:", names)
    print("Ambulance class IDs:", ambulance_ids)

    if not ambulance_ids:
        raise ValueError(
            "Ambulance class not found. "
            "Check your trained model and dataset class names."
        )

    return ambulance_ids


# ==========================================================
# 5. TRAFFIC CLASSIFICATION
# ==========================================================

def get_traffic_level(max_vehicle_count):

    if max_vehicle_count <= LOW_MAX:
        return "LOW"

    elif max_vehicle_count <= MEDIUM_MAX:
        return "MEDIUM"

    return "HIGH"


# ==========================================================
# 6. GREEN TIME ALLOCATION
# ==========================================================

def get_normal_green_time(traffic_level):

    return GREEN_TIME_MAPPING[traffic_level]


# ==========================================================
# 7. VEHICLE DETECTION AND COUNTING
# ==========================================================

def detect_vehicles(vehicle_model, frame):

    results = vehicle_model.predict(
        source=frame,
        conf=VEHICLE_CONFIDENCE,
        verbose=False
    )

    vehicle_count = 0
    vehicle_detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls.item())

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicle_count += 1

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            vehicle_detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_name": vehicle_model.names[class_id]
            })

    return vehicle_count, vehicle_detections


# ==========================================================
# 8. DRAW NORMAL VEHICLE DETECTIONS
# ==========================================================

def draw_vehicle_detections(frame, detections):

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        confidence = detection["confidence"]
        class_name = detection["class_name"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (max(5, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA
        )


# ==========================================================
# 9. AMBULANCE DETECTION
# ==========================================================

def detect_ambulance(
    ambulance_model,
    frame,
    ambulance_class_ids
):

    results = ambulance_model.predict(
        source=frame,
        conf=AMBULANCE_CONFIDENCE,
        imgsz=AMBULANCE_IMAGE_SIZE,
        iou=0.45,
        verbose=False
    )

    best_detection = None
    highest_confidence = 0.0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls.item())

            if class_id not in ambulance_class_ids:
                continue

            confidence = float(box.conf.item())

            if confidence > highest_confidence:

                highest_confidence = confidence

                bbox = tuple(
                    map(int, box.xyxy[0].tolist())
                )

                best_detection = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id
                }

    return best_detection


# ==========================================================
# 10. DRAW AMBULANCE IN RED
# ==========================================================

def draw_ambulance_box(
    frame,
    bbox,
    confidence,
    detected_now=True
):

    x1, y1, x2, y2 = bbox

    height, width = frame.shape[:2]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        RED,
        4
    )

    if detected_now:
        label = f"AMBULANCE DETECTED {confidence:.2f}"
    else:
        label = f"AMBULANCE TRACKING {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    label_y = max(
        0,
        y1 - text_height - 15
    )

    cv2.rectangle(
        frame,
        (x1, label_y),
        (
            min(width - 1, x1 + text_width + 12),
            label_y + text_height + baseline + 10
        ),
        RED,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 6, label_y + text_height + 4),
        font,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA
    )


# ==========================================================
# 11. TRANSPARENT PANEL
# ==========================================================

def draw_panel(
    frame,
    x,
    y,
    width,
    height,
    color=BLACK,
    alpha=0.80
):

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


# ==========================================================
# 12. RIGHT-SIDE TRAFFIC LIGHT
# ==========================================================

def draw_traffic_signal(frame, signal_state):

    height, width = frame.shape[:2]

    box_width = max(65, int(width * 0.065))
    box_height = max(180, int(height * 0.30))

    margin = 15

    x = width - box_width - margin
    y = 20

    center_x = x + box_width // 2

    radius = max(17, int(box_width * 0.28))

    positions = {
        "RED": (center_x, y + 38),
        "YELLOW": (center_x, y + box_height // 2),
        "GREEN": (center_x, y + box_height - 38)
    }

    colors = {
        "RED": RED,
        "YELLOW": YELLOW,
        "GREEN": GREEN
    }

    cv2.rectangle(
        frame,
        (x, y),
        (x + box_width, y + box_height),
        (15, 15, 15),
        -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + box_width, y + box_height),
        (100, 100, 100),
        2
    )

    for state, position in positions.items():

        light_color = (
            colors[state]
            if state == signal_state
            else (55, 55, 55)
        )

        cv2.circle(
            frame,
            position,
            radius,
            light_color,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            position,
            radius,
            (110, 110, 110),
            1,
            cv2.LINE_AA
        )


# ==========================================================
# 13. DASHBOARD
# ==========================================================

def draw_dashboard(
    frame,
    signal_state,
    current_vehicle_count,
    max_vehicle_count,
    traffic_level,
    green_seconds,
    phase_remaining,
    ambulance_priority,
    observation_count,
    observation_frames
):

    height, width = frame.shape[:2]

    # ----------------------------------------------
    # VEHICLE INFORMATION PANEL
    # ----------------------------------------------

    info_x = 15
    info_y = 18

    info_width = min(370, int(width * 0.34))
    info_height = 125

    draw_panel(
        frame,
        info_x,
        info_y,
        info_width,
        info_height,
        GREY,
        0.82
    )

    cv2.putText(
        frame,
        f"LIVE VEHICLES: {current_vehicle_count}",
        (info_x + 10, info_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"MAX VEHICLES: {max_vehicle_count}",
        (info_x + 10, info_y + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"TRAFFIC LEVEL: {traffic_level}",
        (info_x + 10, info_y + 79),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        ORANGE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"GREEN ALLOCATED: {green_seconds} seconds",
        (info_x + 10, info_y + 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        2,
        cv2.LINE_AA
    )

    # ----------------------------------------------
    # TIMER PANEL LEFT OF RIGHT-SIDE TRAFFIC LIGHT
    # ----------------------------------------------

    signal_width = max(65, int(width * 0.065))
    signal_x = width - signal_width - 15

    timer_width = min(
        340,
        max(200, int(width * 0.30))
    )

    timer_height = 130

    timer_x = signal_x - timer_width - 12
    timer_y = 25

    timer_x = max(
        info_x + info_width + 10,
        timer_x
    )

    if timer_x + timer_width > signal_x - 5:
        timer_width = max(
            170,
            signal_x - timer_x - 8
        )

    # ----------------------------------------------
    # TIMER COLOR
    # ----------------------------------------------

    if ambulance_priority:

        timer_color = GREEN
        timer_title = "GREEN TIME REMAINING"

    elif signal_state == "RED":

        timer_color = RED
        timer_title = "RED TIME REMAINING"

    elif signal_state == "YELLOW":

        timer_color = YELLOW
        timer_title = "YELLOW TIME REMAINING"

    else:

        timer_color = GREEN
        timer_title = "GREEN TIME REMAINING"

    draw_panel(
        frame,
        timer_x,
        timer_y,
        timer_width,
        timer_height,
        BLACK,
        0.90
    )

    cv2.rectangle(
        frame,
        (timer_x, timer_y),
        (timer_x + timer_width, timer_y + timer_height),
        timer_color,
        3
    )

    cv2.putText(
        frame,
        timer_title,
        (timer_x + 12, timer_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        timer_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"{phase_remaining:02d}",
        (timer_x + 35, timer_y + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        timer_color,
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "SECONDS",
        (timer_x + 145, timer_y + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        2,
        cv2.LINE_AA
    )

    # ----------------------------------------------
    # AMBULANCE PRIORITY STATUS
    # ----------------------------------------------

    if ambulance_priority:

        cv2.putText(
            frame,
            "AMBULANCE PRIORITY ACTIVE",
            (timer_x, timer_y + timer_height + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            RED,
            2,
            cv2.LINE_AA
        )

    elif signal_state == "RED":

        observation_text = (
            f"OBSERVING: {observation_count}/"
            f"{observation_frames} FRAMES"
        )

        cv2.putText(
            frame,
            observation_text,
            (timer_x, timer_y + timer_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            WHITE,
            1,
            cv2.LINE_AA
        )

    # ----------------------------------------------
    # RIGHT-SIDE TRAFFIC LIGHT
    # ----------------------------------------------

    draw_traffic_signal(
        frame,
        signal_state
    )


# ==========================================================
# 14. MAIN VIDEO PROCESSING
# ==========================================================

def main():

    print("\nSMART TRAFFIC MANAGEMENT SYSTEM")
    print("================================")

    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    if not AMBULANCE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Ambulance model not found: {AMBULANCE_MODEL_PATH}"
        )

    # ----------------------------------------------
    # LOAD AMBULANCE MODEL
    # ----------------------------------------------

    print("\nLoading ambulance detection model...")

    ambulance_model = YOLO(
        str(AMBULANCE_MODEL_PATH)
    )

    ambulance_class_ids = get_ambulance_class_ids(
        ambulance_model
    )

    # ----------------------------------------------
    # LOAD VEHICLE MODEL
    # ----------------------------------------------

    print("\nLoading vehicle detection model...")

    vehicle_model = YOLO(
        VEHICLE_MODEL_PATH
    )

    # ----------------------------------------------
    # OPEN INPUT VIDEO
    # ----------------------------------------------

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    red_observation_frames = max(
        1,
        round(RED_OBSERVATION_SECONDS * fps)
    )

    yellow_frames = max(
        1,
        round(YELLOW_SECONDS * fps)
    )

    priority_frames = max(
        1,
        round(AMBULANCE_PRIORITY_SECONDS * fps)
    )

    persistence_frames = max(
        1,
        round(BOX_PERSISTENCE_SECONDS * fps)
    )

    # ----------------------------------------------
    # CREATE OUTPUT VIDEO
    # ----------------------------------------------

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    # ==================================================
    # NORMAL TRAFFIC STATE
    # ==================================================

    normal_phase = "RED"
    normal_phase_frame = 0

    observation_count = 0
    observation_counts = []

    current_vehicle_count = 0

    max_vehicle_count = 0
    traffic_level = "LOW"

    normal_green_seconds = GREEN_TIME_MAPPING["LOW"]

    # ==================================================
    # AMBULANCE PRIORITY STATE
    # ==================================================

    # This is the remaining number of frames in the
    # ambulance priority countdown.

    ambulance_priority_frames_remaining = 0

    # Rising-edge detection prevents continuous detection
    # from restarting the 20-second timer.

    ambulance_was_detected = False

    last_ambulance_bbox = None
    last_ambulance_confidence = 0.0
    last_ambulance_frame = -1

    frame_count = 0

    print("\nINITIAL SIGNAL: RED")
    print("INITIAL RED COUNTDOWN: 30 SECONDS")
    print("NORMAL YELLOW: 5 SECONDS")
    print("AMBULANCE GREEN PRIORITY: 20 SECONDS")

    print("\nTRAFFIC CLASSIFICATION")
    print("0-5 vehicles   = LOW    = 10 seconds")
    print("6-15 vehicles  = MEDIUM = 20 seconds")
    print("16+ vehicles   = HIGH   = 30 seconds")

    # ==================================================
    # VIDEO PROCESSING LOOP
    # ==================================================

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # ------------------------------------------
            # VEHICLE DETECTION
            # ------------------------------------------

            current_vehicle_count, vehicle_detections = (
                detect_vehicles(
                    vehicle_model,
                    frame
                )
            )

            # ------------------------------------------
            # AMBULANCE DETECTION
            # ------------------------------------------

            ambulance_detection = detect_ambulance(
                ambulance_model,
                frame,
                ambulance_class_ids
            )

            ambulance_detected_now = (
                ambulance_detection is not None
            )

            # ------------------------------------------
            # SAVE LAST AMBULANCE BOX
            # ------------------------------------------

            if ambulance_detected_now:

                last_ambulance_bbox = (
                    ambulance_detection["bbox"]
                )

                last_ambulance_confidence = (
                    ambulance_detection["confidence"]
                )

                last_ambulance_frame = frame_count

            # ------------------------------------------
            # START AMBULANCE PRIORITY
            # ------------------------------------------

            # Start a new 20-second period only when:
            # 1. An ambulance is newly detected.
            # 2. No previous priority countdown is running.
            #
            # Continuous detection cannot restart the timer.

            if (
                ambulance_detected_now
                and not ambulance_was_detected
                and ambulance_priority_frames_remaining <= 0
            ):

                ambulance_priority_frames_remaining = (
                    priority_frames
                )

                print("\nAMBULANCE DETECTED!")
                print("NORMAL SIGNAL INTERRUPTED.")
                print("SIGNAL CHANGED TO GREEN.")
                print("AMBULANCE PRIORITY STARTED: 20 SECONDS.")

            ambulance_was_detected = ambulance_detected_now

            # ------------------------------------------
            # DETERMINE AMBULANCE PRIORITY
            # ------------------------------------------

            ambulance_priority = (
                ambulance_priority_frames_remaining > 0
            )

            # Countdown is calculated before decrementing,
            # so the first ambulance frame displays 20.

            priority_remaining = 0

            if ambulance_priority:

                priority_remaining = math.ceil(
                    ambulance_priority_frames_remaining / fps
                )

            # ------------------------------------------
            # NORMAL TRAFFIC STATE MACHINE
            # ------------------------------------------

            # Normal signal timing is paused while ambulance
            # priority is active. It resumes afterwards.

            if not ambulance_priority:

                # ======================================
                # PHASE 1: RED - 30 SECOND OBSERVATION
                # ======================================

                if normal_phase == "RED":

                    observation_counts.append(
                        current_vehicle_count
                    )

                    observation_count += 1
                    normal_phase_frame += 1

                    # Maximum vehicle count in any frame.
                    max_vehicle_count = max(
                        observation_counts,
                        default=0
                    )

                    # Continuously update classification.
                    traffic_level = get_traffic_level(
                        max_vehicle_count
                    )

                    # Allocation always matches classification.
                    normal_green_seconds = (
                        get_normal_green_time(
                            traffic_level
                        )
                    )

                    if normal_phase_frame >= red_observation_frames:

                        max_vehicle_count = max(
                            observation_counts,
                            default=0
                        )

                        traffic_level = get_traffic_level(
                            max_vehicle_count
                        )

                        normal_green_seconds = (
                            get_normal_green_time(
                                traffic_level
                            )
                        )

                        print("\nTRAFFIC OBSERVATION COMPLETED")
                        print("--------------------------------")

                        print(
                            "Frames observed:",
                            len(observation_counts)
                        )

                        print(
                            "Maximum vehicles:",
                            max_vehicle_count
                        )

                        print(
                            "Traffic level:",
                            traffic_level
                        )

                        print(
                            "Green allocated:",
                            normal_green_seconds,
                            "seconds"
                        )

                        normal_phase = "YELLOW"
                        normal_phase_frame = 0

                # ======================================
                # PHASE 2: YELLOW - 5 SECONDS
                # ======================================

                elif normal_phase == "YELLOW":

                    normal_phase_frame += 1

                    if normal_phase_frame >= yellow_frames:

                        normal_phase = "GREEN"
                        normal_phase_frame = 0

                # ======================================
                # PHASE 3: NORMAL GREEN
                # ======================================

                elif normal_phase == "GREEN":

                    normal_phase_frame += 1

                    normal_green_frames = max(
                        1,
                        round(normal_green_seconds * fps)
                    )

                    if normal_phase_frame >= normal_green_frames:

                        normal_phase = "RED"
                        normal_phase_frame = 0

                        observation_count = 0
                        observation_counts = []

                        max_vehicle_count = 0

            # ------------------------------------------
            # SELECT SIGNAL AND COUNTDOWN
            # ------------------------------------------

            if ambulance_priority:

                signal_state = "GREEN"

                phase_remaining = priority_remaining

                green_seconds = AMBULANCE_PRIORITY_SECONDS

            elif normal_phase == "RED":

                signal_state = "RED"

                phase_remaining = max(
                    0,
                    math.ceil(
                        (
                            red_observation_frames
                            - normal_phase_frame
                        ) / fps
                    )
                )

                green_seconds = normal_green_seconds

            elif normal_phase == "YELLOW":

                signal_state = "YELLOW"

                phase_remaining = max(
                    0,
                    math.ceil(
                        (
                            yellow_frames
                            - normal_phase_frame
                        ) / fps
                    )
                )

                green_seconds = normal_green_seconds

            else:

                signal_state = "GREEN"

                phase_remaining = max(
                    0,
                    normal_green_seconds
                    - math.floor(normal_phase_frame / fps)
                )

                green_seconds = normal_green_seconds

            # ------------------------------------------
            # DRAW NORMAL VEHICLE BOXES
            # ------------------------------------------

            # Other vehicle boxes are hidden during ambulance
            # priority. Vehicle detection/counting continues.

            if not ambulance_priority:

                draw_vehicle_detections(
                    frame,
                    vehicle_detections
                )

            # ------------------------------------------
            # DRAW AMBULANCE RED BOX
            # ------------------------------------------

            if (
                ambulance_priority
                and last_ambulance_bbox is not None
            ):

                frames_since_detection = (
                    frame_count - last_ambulance_frame
                )

                if frames_since_detection <= persistence_frames:

                    draw_ambulance_box(
                        frame,
                        last_ambulance_bbox,
                        last_ambulance_confidence,
                        detected_now=ambulance_detected_now
                    )

            # ------------------------------------------
            # DRAW DASHBOARD
            # ------------------------------------------

            draw_dashboard(
                frame,
                signal_state,
                current_vehicle_count,
                max_vehicle_count,
                traffic_level,
                green_seconds,
                phase_remaining,
                ambulance_priority,
                observation_count,
                red_observation_frames
            )

            # ------------------------------------------
            # SAVE OUTPUT FRAME
            # ------------------------------------------

            writer.write(frame)

            # ------------------------------------------
            # DISPLAY OUTPUT
            # ------------------------------------------

            cv2.imshow(
                "Smart Traffic Management System",
                frame
            )

            # ------------------------------------------
            # DECREASE AMBULANCE TIMER
            # ------------------------------------------

            # This is deliberately outside the detection
            # condition. It decreases every processed frame,
            # even when ambulance_detected_now is False.

            if ambulance_priority_frames_remaining > 0:

                ambulance_priority_frames_remaining -= 1

                if ambulance_priority_frames_remaining == 0:

                    print("\nAMBULANCE PRIORITY COMPLETED.")
                    print("20-SECOND COUNTDOWN FINISHED.")
                    print("RESUMING NORMAL TRAFFIC SIGNAL.")

            # ------------------------------------------
            # EXIT OPTION
            # ------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):

                print("Processing stopped by user.")
                break

            # ------------------------------------------
            # TERMINAL PROGRESS
            # ------------------------------------------

            if frame_count % 30 == 0:

                print(
                    f"Frame: {frame_count}/{total_frames} | "
                    f"Live: {current_vehicle_count} | "
                    f"Maximum: {max_vehicle_count} | "
                    f"Traffic: {traffic_level} | "
                    f"Ambulance detected: {ambulance_detected_now} | "
                    f"Priority active: {ambulance_priority} | "
                    f"Signal: {signal_state} | "
                    f"Countdown: {phase_remaining}"
                )

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\nPROCESSING COMPLETED")
    print("--------------------------------")
    print("Processed frames:", frame_count)
    print("Output saved at:", OUTPUT_VIDEO)


# ==========================================================
# 15. PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
