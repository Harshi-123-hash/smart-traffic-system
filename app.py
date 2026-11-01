
import os
import sys
import sqlite3
import threading
import subprocess
import json
import time
from datetime import datetime

from flask import Flask, render_template, jsonify, request, redirect, url_for

# Optional ambulance simulation module
try:
    from src.ambulance_dispatch import dispatch_ambulance
except Exception as exc:
    print(f"Ambulance module unavailable: {exc}")
    dispatch_ambulance = None


app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "traffic_events.db")
DETECTION_SCRIPT = os.path.join(
    PROJECT_DIR, "src", "accident_response.py"
)
VEHICLE_SCRIPT = os.path.join(PROJECT_DIR, "src", "vehicle_count.py")
VEHICLE_STATUS_PATH = os.path.join(PROJECT_DIR, "vehicle_status.json")
VEHICLE_LOG_PATH = os.path.join(PROJECT_DIR, "results", "vehicle_count_log.txt")

# Example coordinates for simulation only
DEFAULT_LATITUDE = 16.5080
DEFAULT_LONGITUDE = 80.6400

state_lock = threading.RLock()
detection_lock = threading.Lock()

detection_process = None
detection_thread = None
vehicle_process = None
vehicle_lock = threading.Lock()
vehicle_loop_enabled = False
vehicle_loop_thread = None
vehicle_loop_stop = threading.Event()
detection_output = []
alert_ids = set()

system_state = {
    "traffic_density": "LOW",
    "vehicle_count": 0,
    "signal_phase": "NORMAL",
    "green_seconds": 10,
    "ambulance_status": "Available (simulated)",
    "ambulance_traffic": "N/A",
    "ambulance_eta_minutes": None,
    "ambulance_distance": "N/A",
    "incident_status": "No incident reported",
}


# --------------------------------------------------
# Database functions
# --------------------------------------------------

def get_db():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the events table without deleting existing event history."""

    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'Incident',
                status TEXT NOT NULL DEFAULT 'Unknown',
                latitude REAL,
                longitude REAL,
                ambulance TEXT DEFAULT 'N/A',
                distance TEXT DEFAULT 'N/A',
                distance_km REAL,
                dispatch TEXT DEFAULT 'Simulated',
                ambulance_traffic TEXT DEFAULT 'N/A',
                ambulance_eta_minutes REAL,
                traffic_density TEXT DEFAULT 'LOW',
                vehicle_count INTEGER DEFAULT 0,
                signal_phase TEXT DEFAULT 'NORMAL',
                green_seconds INTEGER DEFAULT 10,
                details TEXT DEFAULT ''
            )
        """)

        # Add missing columns if an older database already exists.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(events)"
            ).fetchall()
        }

        required_columns = {
            "event_type": "TEXT DEFAULT 'Incident'",
            "status": "TEXT DEFAULT 'Unknown'",
            "latitude": "REAL",
            "longitude": "REAL",
            "ambulance": "TEXT DEFAULT 'N/A'",
            "distance": "TEXT DEFAULT 'N/A'",
            "distance_km": "REAL",
            "dispatch": "TEXT DEFAULT 'Simulated'",
            "ambulance_traffic": "TEXT DEFAULT 'N/A'",
            "ambulance_eta_minutes": "REAL",
            "traffic_density": "TEXT DEFAULT 'LOW'",
            "vehicle_count": "INTEGER DEFAULT 0",
            "signal_phase": "TEXT DEFAULT 'NORMAL'",
            "green_seconds": "INTEGER DEFAULT 10",
            "details": "TEXT DEFAULT ''",
        }

        for column, definition in required_columns.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE events ADD COLUMN "
                    f"{column} {definition}"
                )


def row_to_dict(row):
    return dict(row) if row is not None else None


def get_events(limit=100):
    with get_db() as connection:
        rows = connection.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return [row_to_dict(row) for row in rows]


def record_event(event):
    """Insert an event into SQLite and return its database ID."""

    fields = [
        "time",
        "event_type",
        "status",
        "latitude",
        "longitude",
        "ambulance",
        "distance",
        "distance_km",
        "dispatch",
        "ambulance_traffic",
        "ambulance_eta_minutes",
        "traffic_density",
        "vehicle_count",
        "signal_phase",
        "green_seconds",
        "details",
    ]

    values = [event.get(field) for field in fields]

    if not values[0]:
        values[0] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    placeholders = ", ".join("?" for _ in fields)

    with get_db() as connection:
        cursor = connection.execute(
            f"INSERT INTO events ({', '.join(fields)}) "
            f"VALUES ({placeholders})",
            values
        )

        return cursor.lastrowid


def restore_state_from_latest_event():
    """Restore the dashboard state from the most recent event."""

    try:
        with get_db() as connection:
            row = connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT 1"
            ).fetchone()

        if row is None:
            return

        event = row_to_dict(row)

        with state_lock:
            for key in (
                "traffic_density",
                "vehicle_count",
                "signal_phase",
                "green_seconds",
            ):
                if event.get(key) is not None:
                    system_state[key] = event[key]

            if event.get("status"):
                system_state["incident_status"] = event["status"]

            ambulance = event.get("ambulance")
            if ambulance and ambulance != "N/A":
                system_state["ambulance_status"] = ambulance
            if "ambulance_traffic" in event:
                system_state["ambulance_traffic"] = event.get("ambulance_traffic") or "N/A"
            if "ambulance_eta_minutes" in event:
                system_state["ambulance_eta_minutes"] = event.get("ambulance_eta_minutes")
            if event.get("distance"):
                system_state["ambulance_distance"] = event.get("distance")

    except sqlite3.Error as exc:
        print(f"Could not restore previous state: {exc}")


# --------------------------------------------------
# Ambulance simulation
# --------------------------------------------------


def get_ambulance_details(latitude, longitude, traffic_level="MEDIUM"):
    """Return ambulance details including estimated traffic and ETA."""

    if dispatch_ambulance is None:
        return {
            "ambulance": "AMB-01 (simulated)",
            "distance": "N/A",
            "distance_km": None,
            "dispatch": "Simulation only",
            "traffic_level": "N/A",
            "estimated_eta_minutes": None,
        }

    try:
        result = dispatch_ambulance(
    latitude,
    longitude,
    traffic_level=traffic_level
)

        if isinstance(result, dict):
            ambulance = (
                result.get("ambulance")
                or result.get("ambulance_id")
                or result.get("vehicle")
                or result.get("id")
                or result.get("name")
                or "AMB-01 (simulated)"
            )

            raw_distance = result.get(
                "distance_km",
                result.get("distance")
            )

            try:
                distance_km = (
                    float(raw_distance)
                    if raw_distance is not None
                    else None
                )
            except (TypeError, ValueError):
                distance_km = None

            distance_text = (
                f"{distance_km:.2f} km"
                if distance_km is not None
                else str(raw_distance or "N/A")
            )

            return {
                "ambulance": str(ambulance),
                "distance": distance_text,
                "distance_km": distance_km,
                "dispatch": str(
                    result.get(
                        "dispatch",
                        result.get("status", "Simulated dispatch")
                    )
                ),
                "traffic_level": result.get("traffic_level", "N/A"),
                "estimated_eta_minutes": result.get(
                    "estimated_eta_minutes"
                ),
            }

        if isinstance(result, (tuple, list)):
            ambulance = (
                str(result[0])
                if len(result) > 0
                else "AMB-01 (simulated)"
            )
            raw_distance = result[1] if len(result) > 1 else None

            try:
                distance_km = (
                    float(raw_distance)
                    if raw_distance is not None
                    else None
                )
            except (TypeError, ValueError):
                distance_km = None

            return {
                "ambulance": ambulance,
                "distance": (
                    f"{distance_km:.2f} km"
                    if distance_km is not None
                    else str(raw_distance or "N/A")
                ),
                "distance_km": distance_km,
                "dispatch": "Simulated dispatch",
                "traffic_level": "N/A",
                "estimated_eta_minutes": None,
            }

        return {
            "ambulance": str(result or "AMB-01 (simulated)"),
            "distance": "N/A",
            "distance_km": None,
            "dispatch": "Simulation only",
            "traffic_level": "N/A",
            "estimated_eta_minutes": None,
        }

    except Exception as exc:
        print(f"Ambulance simulation warning: {exc}")

        return {
            "ambulance": "AMB-01 (simulated)",
            "distance": "N/A",
            "distance_km": None,
            "dispatch": "Simulation fallback",
            "traffic_level": "N/A",
            "estimated_eta_minutes": None,
        }


# --------------------------------------------------
# Vehicle status helpers
# --------------------------------------------------

def read_vehicle_status():
    """Read the latest status written by the vehicle-counting process."""
    default_status = {
        "running": False,
        "vehicle_count": 0,
        "traffic_density": "UNKNOWN",
        "last_frame": 0,
        "processed_frames": 0,
        "message": "Vehicle counting has not started",
    }
    try:
        with open(VEHICLE_STATUS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return default_status
        return {**default_status, **data}
    except (OSError, ValueError):
        return default_status



def sync_vehicle_state():
    """Read the latest vehicle count and calculate traffic density."""
    status = read_vehicle_status()

    count = status.get("vehicle_count")

    try:
        count = int(count)
    except (TypeError, ValueError):
        with state_lock:
            count = system_state.get("vehicle_count", 0)

    if count < 0:
        count = 0

    if count <= 5:
        density = "LOW"
    elif count <= 15:
        density = "MEDIUM"
    else:
        density = "HIGH"

    with state_lock:
        system_state["vehicle_count"] = count
        system_state["traffic_density"] = density

    status["vehicle_count"] = count
    status["traffic_density"] = density

    return status

# --------------------------------------------------
# Accident alert creation
# --------------------------------------------------


def create_accident_alert(
    details="Possible accident - verify",
    event_type="Accident Alert"
):
    """Record an accident and select an ambulance using current traffic."""
    # Pass the traffic level to the ambulance dispatch module.
    sync_vehicle_state()

    with state_lock:
        traffic_level = system_state["traffic_density"]
        vehicle_count = system_state["vehicle_count"]

    ambulance_info = get_ambulance_details(
        DEFAULT_LATITUDE,
        DEFAULT_LONGITUDE,
        traffic_level
    )

    density_green_seconds = {
        "LOW": 10,
        "MEDIUM": 20,
        "HIGH": 30
    }.get(traffic_level, 20)

    with state_lock:
        event = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "status": details,
            "latitude": DEFAULT_LATITUDE,
            "longitude": DEFAULT_LONGITUDE,
            "ambulance": ambulance_info["ambulance"],
            "distance": ambulance_info["distance"],
            "distance_km": ambulance_info["distance_km"],
            "dispatch": ambulance_info["dispatch"],
            "ambulance_traffic": ambulance_info.get(
                "traffic_level", "N/A"
            ),
            "ambulance_eta_minutes": ambulance_info.get(
                "estimated_eta_minutes"
            ),
            "traffic_density": traffic_level,
            "vehicle_count": vehicle_count,
            "signal_phase": "AMBULANCE PRIORITY",
            "green_seconds": density_green_seconds,
            "details": details,
        }

        event_id = record_event(event)
        event["id"] = event_id

        system_state["incident_status"] = details
        system_state["signal_phase"] = "AMBULANCE PRIORITY"
        system_state["green_seconds"] = density_green_seconds
        system_state["ambulance_status"] = ambulance_info["ambulance"]
        system_state["ambulance_traffic"] = ambulance_info.get("traffic_level", "N/A")
        system_state["ambulance_eta_minutes"] = ambulance_info.get("estimated_eta_minutes")
        system_state["ambulance_distance"] = ambulance_info.get("distance", "N/A")

    print(f"Accident alert recorded. Event ID: {event_id}")
    return event


# --------------------------------------------------
# Dashboard routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    with state_lock:
        current_state = dict(system_state)
        current_output = list(detection_output)

    events = get_events()

    return render_template(
        "index.html",
        state=current_state,
        alerts=events,
        total_alerts=len(events),
        detection_output=current_output,
    )


@app.route("/api/dashboard-state", methods=["GET"])
def dashboard_state():
    events = get_events()

    with state_lock:
        current_state = dict(system_state)
        current_output = list(detection_output)

    return jsonify({
        "state": current_state,
        "events": events,
        "total_alerts": len(events),
        "detection_output": current_output,
    })


@app.route("/simulate-alert", methods=["POST"])
def simulate_alert():
    details = ""

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        details = str(payload.get("details", "")).strip()
    else:
        details = request.form.get("details", "").strip()

    if not details:
        details = "Manually simulated accident alert"

    event = create_accident_alert(
        details=details,
        event_type="Simulated Accident"
    )

    wants_json = (
        request.is_json
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )

    if wants_json:
        return jsonify({
            "success": True,
            "message": "Simulated accident alert recorded.",
            "event": event,
        }), 201

    return redirect(url_for("index"))


@app.route("/api/incident", methods=["GET", "POST"])
def incident():
    if request.method == "GET":
        with state_lock:
            return jsonify({
                "incident_status": system_state["incident_status"],
                "signal_phase": system_state["signal_phase"],
                "green_seconds": system_state["green_seconds"],
            })

    payload = request.get_json(silent=True)

    if payload is None:
        payload = request.form.to_dict()

    details = (
        payload.get("details")
        or payload.get("status")
        or "Possible accident - verify"
    )

    event = create_accident_alert(
        details=str(details),
        event_type="Accident Alert"
    )

    return jsonify({
        "success": True,
        "event": event,
    }), 201


# --------------------------------------------------
# Video accident detector
# --------------------------------------------------

def _read_detection_output(process):
    """Read detector output and create alerts when an accident is reported."""

    global detection_process

    try:
        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()

                if not line:
                    continue

                with state_lock:
                    detection_output.append(line)

                    if len(detection_output) > 300:
                        del detection_output[:-300]

                print(f"[detector] {line}")

                # The detector must print this phrase to trigger an alert.
                if "ACCIDENT DETECTED AT FRAME" in line.upper():
                    marker = line.upper()

                    with detection_lock:
                        is_new_alert = marker not in alert_ids
                        if is_new_alert:
                            alert_ids.add(marker)

                    if is_new_alert:
                        create_accident_alert(
                            details=f"Video detector: {line}",
                            event_type="Video Accident Detection"
                        )

        return_code = process.wait()

        with state_lock:
            detection_output.append(
                f"Detection process finished (exit code {return_code})."
            )

    except Exception as exc:
        with state_lock:
            detection_output.append(
                f"Detection reader error: {exc}"
            )

    finally:
        if process.stdout:
            process.stdout.close()

        with detection_lock:
            if detection_process is process:
                detection_process = None


@app.route("/start-detection", methods=["POST"])
def start_detection():
    global detection_process, detection_thread

    with detection_lock:
        if (
            detection_process is not None
            and detection_process.poll() is None
        ):
            return jsonify({
                "success": True,
                "message": "Accident detection is already running.",
            })

        if not os.path.isfile(DETECTION_SCRIPT):
            message = (
                "Detection script not found: "
                + DETECTION_SCRIPT
            )

            with state_lock:
                detection_output.append(message)

            return jsonify({
                "success": False,
                "message": message,
            }), 404

        with state_lock:
            detection_output.clear()
            detection_output.append(
                "Starting accident detection script..."
            )

        alert_ids.clear()

        try:
            detection_process = subprocess.Popen(
                [sys.executable, "-u", DETECTION_SCRIPT],
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            detection_thread = threading.Thread(
                target=_read_detection_output,
                args=(detection_process,),
                daemon=True,
            )

            detection_thread.start()

            return jsonify({
                "success": True,
                "message": "Accident detection started.",
            })

        except Exception as exc:
            detection_process = None
            message = f"Could not start detection: {exc}"

            with state_lock:
                detection_output.append(message)

            return jsonify({
                "success": False,
                "message": message,
            }), 500


@app.route("/detection-status", methods=["GET"])
def detection_status():
    with state_lock:
        output = list(detection_output)

    process = detection_process
    running = (
        process is not None
        and process.poll() is None
    )

    return jsonify({
        "running": running,
        "output": output,
    })


# --------------------------------------------------
# Vehicle counting routes
# --------------------------------------------------

@app.route("/api/vehicle-status", methods=["GET"])
def vehicle_status():
    status = sync_vehicle_state()
    return jsonify(status)


def run_vehicle_count_loop():
    """Run vehicle counting repeatedly until the stop endpoint is called."""
    global vehicle_process

    while not vehicle_loop_stop.is_set():
        process = None

        try:
            os.makedirs(os.path.dirname(VEHICLE_LOG_PATH), exist_ok=True)

            with open(VEHICLE_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write("\nStarting a new vehicle-counting cycle...\n")
                log_file.flush()

                if vehicle_loop_stop.is_set():
                    break

                process = subprocess.Popen(
                    [sys.executable, "-u", VEHICLE_SCRIPT],
                    cwd=PROJECT_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

                with vehicle_lock:
                    vehicle_process = process

                return_code = process.wait()
                log_file.write(
                    f"Vehicle-counting cycle finished with exit code {return_code}.\n"
                )
                log_file.flush()

        except Exception as exc:
            print(f"Vehicle-counting cycle error: {exc}")

        finally:
            with vehicle_lock:
                if vehicle_process is process:
                    vehicle_process = None

        if vehicle_loop_stop.wait(5):
            break


@app.route("/start-vehicle-count", methods=["POST"])
def start_vehicle_count():
    global vehicle_loop_enabled, vehicle_loop_thread

    if not os.path.isfile(VEHICLE_SCRIPT):
        return jsonify({
            "success": False,
            "message": f"Vehicle-counting script not found: {VEHICLE_SCRIPT}",
        }), 404

    with vehicle_lock:
        if vehicle_loop_enabled:
            return jsonify({
                "success": True,
                "message": "Automatic vehicle analysis is already running.",
            }), 200

        vehicle_loop_stop.clear()
        vehicle_loop_enabled = True
        vehicle_loop_thread = threading.Thread(
            target=run_vehicle_count_loop,
            daemon=True,
            name="vehicle-count-loop",
        )
        vehicle_loop_thread.start()

    return jsonify({
        "success": True,
        "message": "Automatic repeated vehicle analysis started.",
    }), 202


@app.route("/stop-vehicle-count", methods=["POST"])
def stop_vehicle_count():
    global vehicle_loop_enabled, vehicle_process

    with vehicle_lock:
        vehicle_loop_enabled = False
        vehicle_loop_stop.set()
        process = vehicle_process

        if process is not None and process.poll() is None:
            process.terminate()

    return jsonify({
        "success": True,
        "message": "Automatic vehicle analysis is stopping.",
    }), 200


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "database": os.path.basename(DB_PATH),
    })


# --------------------------------------------------
# Initialize and run
# --------------------------------------------------

initialize_database()
restore_state_from_latest_event()

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )


