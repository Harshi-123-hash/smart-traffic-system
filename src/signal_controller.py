from dataclasses import dataclass


@dataclass(frozen=True)
class SignalDecision:
    traffic_level: str
    green_seconds: int
    reason: str
    ambulance_priority: bool = False


GREEN_TIMES = {
    "LOW": 10,
    "MEDIUM": 20,
    "HIGH": 30,
}

AMBULANCE_GREEN_SECONDS = 30


def decide_green_time(traffic_level, ambulance_detected=False):
    level = traffic_level.strip().upper()

    if level not in GREEN_TIMES:
        raise ValueError(
            f"Unsupported traffic level: {traffic_level}"
        )

    if ambulance_detected:
        return SignalDecision(
            traffic_level=level,
            green_seconds=AMBULANCE_GREEN_SECONDS,
            reason="Ambulance detected. Emergency priority activated.",
            ambulance_priority=True,
        )

    green_seconds = GREEN_TIMES[level]

    return SignalDecision(
        traffic_level=level,
        green_seconds=green_seconds,
        reason=f"{level} traffic receives {green_seconds} seconds of green time.",
        ambulance_priority=False,
    )


if __name__ == "__main__":
    for level in ("LOW", "MEDIUM", "HIGH"):
        decision = decide_green_time(level)
        print(f"Traffic: {decision.traffic_level}")
        print(f"Green time: {decision.green_seconds} seconds")
        print(f"Decision: {decision.reason}")
        print("-" * 40)

    decision = decide_green_time("HIGH", ambulance_detected=True)
    print("AMBULANCE PRIORITY TEST")
    print(f"Green time: {decision.green_seconds} seconds")
    print(f"Priority: {decision.ambulance_priority}")
    print(decision.reason)
