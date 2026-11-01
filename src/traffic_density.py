
from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficDensity:
    vehicle_count: int
    level: str


def estimate_traffic_density(vehicle_count: int) -> TrafficDensity:
    if vehicle_count < 0:
        raise ValueError("Vehicle count cannot be negative.")

    if vehicle_count <= 5:
        level = "LOW"
    elif vehicle_count <= 12:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return TrafficDensity(
        vehicle_count=vehicle_count,
        level=level,
    )


if __name__ == "__main__":
    for count in (3, 8, 16):
        density = estimate_traffic_density(count)
        print(
            f"Vehicles: {density.vehicle_count} | "
            f"Traffic density: {density.level}"
        )
