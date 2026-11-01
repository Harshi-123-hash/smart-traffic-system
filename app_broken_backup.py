import os
import sys
import json
import sqlite3
import threading
import subprocess
from datetime import datetime

from flask import Flask, render_template, jsonify, request, redirect, url_for

try:
    from src.ambulance_dispatch import dispatch_ambulance
except Exception:
    dispatch_ambulance = None

app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "traffic_events.db")
DETECTION_SCRIPT = os.path.join(PROJECT_DIR, "src", "accident_response.py")

# Example coordinates for simulation only; not live GPS.
DEFAULT_LATITUDE = 16.5080
DEFAULT_LONGITUDE = 80.6400

state_lock = threading.RLock()
detection_lock = threading.Lock()
detection_process = None
detection_thread = None
detection_output = []
alert_ids = set()

system_state = {
    "traffic_density": "LOW",
    "vehicle_count": 0,
    "signal_phase": "NORMAL",
    "green_seconds": 10,
    "ambulance_status": "Available (simulated)",
    "incident_status": "No incident reported",
}


# --------------------------------------------------
# Database helpers
# --------------------------------------------------
def get_db():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the events table and add missing columns without deleting data."""
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
                traffic_density TEXT DEFAULT 'LOW',
                vehicle_count INTEGER DEFAULT 0,
                signal_phase TEXT DEFAULT 'NORMAL',
                green_seconds INTEGER DEFAULT 10,
                details TEXT DEFAULT ''
            )
        """)

        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(events)").fetchall()
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
            "traffic_density": "TEXT DEFAULT 'LOW'",
            "vehicle_count": "INTEGER DEFAULT 0",
            "signal_phase": "TEXT DEFAULT 'NORMAL'",
            "green_seconds": "INTEGER DEFAULT 10",
            "details": "TEXT DEFAULT ''",
        }
