
import math
from datetime import datetime

AMBULANCES = [
    {
        "id": "AMB-01",
        "latitude": 16.5062,
        "longitude": 80.6480,
        "available": True,
    },
    {
        "id": "AMB-02",
        "latitude": 16.5100,
        "longitude": 80.6300,
        "available": True,
    },
    {
        "id": "AMB-03",
        "latitude": 16.4900,
        "longitude": 80.6500,
        "available": False,
    },
]


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two coordinates."""
    radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    # Protect against floating-point rounding errors.
    a = max(0.0, min(1.0, a))

    return radius * 2 * math.atan2(
        math.sqrt(a), math.sqrt(1 - a)
    )





def find_nearest_ambulance(
    accident_lat,
    accident_lon,
    traffic_level="MEDIUM"
):
    """Select an available ambulance using traffic-adjusted estimated ETA."""

    available = [
        ambulance.copy()
        for ambulance in AMBULANCES
        if ambulance["available"]
    ]

    if not available:
        return None

    traffic_level = str(traffic_level).upper()

    if traffic_level not in {"LOW", "MEDIUM", "HIGH"}:
        traffic_level = "MEDIUM"

    average_speed_kmph = {
        "LOW": 35,
        "MEDIUM": 20,
        "HIGH": 10,
    }[traffic_level]

    candidates = []

    for ambulance in available:
        distance = calculate_distance_km(
            accident_lat,
            accident_lon,
            ambulance["latitude"],
            ambulance["longitude"],
        )

        eta_minutes = (distance / average_speed_kmph) * 60

        ambulance["distance_km"] = round(distance, 2)
        ambulance["traffic_level"] = traffic_level
        ambulance["estimated_eta_minutes"] = round(eta_minutes, 1)

        candidates.append(ambulance)

    return min(
        candidates,
        key=lambda item: item["estimated_eta_minutes"]
    )


def dispatch_ambulance(
    accident_lat,
    accident_lon,
    traffic_level="MEDIUM"
):
    """Simulate ambulance selection using traffic-adjusted ETA."""

    ambulance = find_nearest_ambulance(
        accident_lat,
        accident_lon,
        traffic_level
    )

    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accident_latitude": accident_lat,
        "accident_longitude": accident_lon,
        "status": "Possible accident - verify",
        "dispatch": "Simulated",
        "id": "Unavailable",
        "distance_km": None,
        "traffic_level": traffic_level,
        "estimated_eta_minutes": None,
        "available": False,
        "message": "No available ambulance found.",
    }

    if ambulance:
        result.update({
            "id": ambulance["id"],
            "distance_km": ambulance["distance_km"],
            "traffic_level": ambulance["traffic_level"],
            "estimated_eta_minutes": ambulance["estimated_eta_minutes"],
            "available": True,
            "message": (
                f"Recommended ambulance: {ambulance['id']} "
                f"with estimated arrival in "
                f"{ambulance['estimated_eta_minutes']} minutes."
            ),
        })

    print("\n========== AMBULANCE DISPATCH ==========")
    print("Time:", result["time"])
    print("Accident location:", accident_lat, accident_lon)
    print("Ambulance:", result["id"])
    print("Distance (km):", result["distance_km"])
    print("Traffic:", result["traffic_level"])
    print("Estimated ETA (minutes):", result["estimated_eta_minutes"])
    print("Dispatch:", result["dispatch"])
    print("========================================")

    return result

if __name__ == "__main__":
    # Example coordinates for testing only.
    dispatch_ambulance(16.5080, 80.6400)