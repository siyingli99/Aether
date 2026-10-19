"""``events.csv``: collisions, off-road, off-scene and run termination."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_events(path: str | Path, events: list[dict]) -> None:
    """``events`` are :meth:`aether.sim.collision.Event.to_dict` records."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "step", "type", "other_id", "at_fault", "detail"])
        for event in events:
            detail = {k: v for k, v in event.items() if k not in ("type", "time", "step")}
            writer.writerow([f'{event["time"]:.3f}', event["step"], event["type"],
                             detail.pop("other", ""), detail.pop("at_fault", ""),
                             json.dumps(detail, ensure_ascii=False)])
