#!/usr/bin/env python3
"""fever_estimator.py - secondary fever-decision stage.

The YOLO model only outputs a species/class label per detection (person,
dog, cat, ...). This module is the second stage: for each detection it asks
thermalCam-Pi (via ThermalCamClient) for the estimated temperature at that
point in the frame, then decides whether that species reading is feverish
using per-species normal ranges. No temperature math happens here — that's
thermalCam-Pi's job; this module only interprets its numbers.
"""

import time
from threading import RLock

# --- Species normal temperature ranges (°C) — estimates only ---
ANIMAL_TEMP_RANGES = {
    "person": (36.5, 37.5),
    "dog": (38.3, 39.2),
    "cat": (38.1, 39.2),
    "cow": (38.0, 39.3),
    "chicken": (40.6, 41.7),
    "sheep": (38.3, 39.9),
    "horse": (37.5, 38.5),
    "goat": (38.5, 39.7),
}

DEFAULT_FEVER_MARGIN_C = 1.0  # °C above species normal-max before flagging fever
DEFAULT_QUERY_INTERVAL = 0.3  # min seconds between /pixel_temp calls to thermalCam-Pi
DEFAULT_MAX_QUERIES_PER_CYCLE = 3  # highest-confidence detections queried per cycle


class FeverEstimator:
    """Turns species-only detections into temp + fever verdicts via thermalCam-Pi."""

    def __init__(self, client, fever_margin=DEFAULT_FEVER_MARGIN_C, temp_ranges=None,
                 query_interval=DEFAULT_QUERY_INTERVAL,
                 max_queries_per_cycle=DEFAULT_MAX_QUERIES_PER_CYCLE):
        self.client = client
        self.fever_margin = fever_margin
        self.temp_ranges = temp_ranges or ANIMAL_TEMP_RANGES
        self.query_interval = query_interval
        self.max_queries_per_cycle = max_queries_per_cycle

        self._lock = RLock()
        self._last_query_ts = 0.0

    def temp_range_for(self, species: str):
        return self.temp_ranges.get((species or "").lower())

    def is_feverish(self, species: str, est_temp: float) -> bool:
        rng = self.temp_range_for(species)
        if rng is None or est_temp is None:
            return False  # unknown species: no trained normal range, no verdict
        _, normal_max = rng
        return est_temp > (normal_max + self.fever_margin)

    def evaluate(self, detections, frame_w: int, frame_h: int):
        """Attach 'temp' and 'fever' fields to species-only detections.

        Only the top `max_queries_per_cycle` (by confidence) detections are
        queried per call, and only if `query_interval` has elapsed since the
        last query batch — this keeps thermalCam-Pi from being hammered with
        one HTTP round trip per detection per frame.
        """
        if not detections or not frame_w or not frame_h:
            return [dict(d, temp=None, fever=None) for d in detections]

        now = time.time()
        with self._lock:
            can_query = (now - self._last_query_ts) >= self.query_interval

        temps = {}
        if can_query:
            ordered = sorted(detections, key=lambda d: d["conf"], reverse=True)
            queried = 0
            for d in ordered:
                if queried >= self.max_queries_per_cycle:
                    break
                xn = d["cx"] / float(frame_w)
                yn = d["cy"] / float(frame_h)
                temp = self.client.get_pixel_temp(xn, yn)
                temps[id(d)] = temp
                queried += 1
            if queried:
                with self._lock:
                    self._last_query_ts = now

        out = []
        for d in detections:
            t = temps.get(id(d))
            item = dict(d)
            item["temp"] = round(t, 1) if t is not None else None
            item["fever"] = self.is_feverish(d["cls"], t) if t is not None else None
            out.append(item)
        return out

    def status(self):
        return {
            "fever_margin": self.fever_margin,
            "query_interval": self.query_interval,
            "max_queries_per_cycle": self.max_queries_per_cycle,
            "known_species": sorted(self.temp_ranges.keys()),
        }
