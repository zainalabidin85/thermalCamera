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
from concurrent.futures import ThreadPoolExecutor
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
DEFAULT_TEMP_HOLD_SECONDS = 1.5  # how long a stale temp reading stays displayed
TEMP_CACHE_MATCH_DIST = 0.08  # normalized (0-1) center-distance to treat as "same" detection

# Coarse 3x3 search grid spread across a detection's bbox (as fractions of
# bbox width/height, inset from the edges to avoid background bleed) -
# queried in one parallel batch just to locate the hottest area.
# thermalCam-Pi's /pixel_temp reads a real per-pixel grayscale value from
# the raw sensor frame (not just distance-from-center extrapolation), so
# different points in the bbox genuinely can read different temps - this
# searches for the hottest spot (skin/fur vs. clothing/background) instead
# of blindly averaging the geometric center.
TEMP_GRID_FRACS = [0.2, 0.5, 0.8]

# Once the hottest grid point is found, refine with 4 more points tightly
# clustered within this many pixels of it (a small "+" pattern), and
# average all 5 - this is the actual reported temperature.
TEMP_REFINE_RADIUS_PX = 4


class FeverEstimator:
    """Turns species-only detections into temp + fever verdicts via thermalCam-Pi."""

    def __init__(self, client, fever_margin=DEFAULT_FEVER_MARGIN_C, temp_ranges=None,
                 query_interval=DEFAULT_QUERY_INTERVAL,
                 max_queries_per_cycle=DEFAULT_MAX_QUERIES_PER_CYCLE,
                 temp_hold_seconds=DEFAULT_TEMP_HOLD_SECONDS):
        self.client = client
        self.fever_margin = fever_margin
        self.temp_ranges = temp_ranges or ANIMAL_TEMP_RANGES
        self.query_interval = query_interval
        self.max_queries_per_cycle = max_queries_per_cycle
        self.temp_hold_seconds = temp_hold_seconds

        self._lock = RLock()
        self._last_query_ts = 0.0
        self._temp_cache = []  # [{"cls","xn","yn","temp","ts"}, ...] most-recent-first
        # sized for the coarse-grid batch across every queried detection at once
        # (that's the larger of the two batches - refine is 4/detection vs. 9)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_queries_per_cycle) * len(TEMP_GRID_FRACS) ** 2)

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
            to_query = ordered[:self.max_queries_per_cycle]
            if to_query:
                temps = self._sample_detections(to_query, frame_w, frame_h)
                for d in to_query:
                    t = temps.get(id(d))
                    if t is not None:
                        xn = d["cx"] / float(frame_w)
                        yn = d["cy"] / float(frame_h)
                        self._cache_temp(d["cls"], xn, yn, t, now)
                with self._lock:
                    self._last_query_ts = now

        out = []
        for d in detections:
            t = temps.get(id(d))
            if t is None:
                xn = d["cx"] / float(frame_w)
                yn = d["cy"] / float(frame_h)
                t = self._lookup_cached_temp(d["cls"], xn, yn, now)
            item = dict(d)
            item["temp"] = round(t, 1) if t is not None else None
            item["fever"] = self.is_feverish(d["cls"], t) if t is not None else None
            out.append(item)
        return out

    def _query_batch(self, points_px, frame_w, frame_h):
        """Fire every point in one parallel batch; returns readings in the same order."""
        return list(self._pool.map(
            lambda p: self.client.get_pixel_temp(p[0] / frame_w, p[1] / frame_h), points_px))

    def _sample_detections(self, dets, frame_w, frame_h):
        """Coarse 3x3 grid search across each bbox to locate its hottest area,
        then refine with 4 tightly-clustered points around it and average all
        5 - two parallel batches total (all detections' grid points together,
        then all detections' refine points together), not two per detection."""
        grids = {}
        grid_px, grid_owner = [], []
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            w, h = (x2 - x1), (y2 - y1)
            pts = [(min(max(x1 + fx * w, 0), frame_w - 1), min(max(y1 + fy * h, 0), frame_h - 1))
                   for fy in TEMP_GRID_FRACS for fx in TEMP_GRID_FRACS]
            grids[id(d)] = pts
            grid_px.extend(pts)
            grid_owner.extend([id(d)] * len(pts))

        grid_readings = self._query_batch(grid_px, frame_w, frame_h)

        hotspots = {}  # id(d) -> ((hot_x, hot_y), hot_t)
        by_det = {}
        for owner, p, t in zip(grid_owner, grid_px, grid_readings):
            if t is not None:
                by_det.setdefault(owner, []).append((p, t))
        for did, valid in by_det.items():
            hotspots[did] = max(valid, key=lambda pt: pt[1])

        r = TEMP_REFINE_RADIUS_PX
        refine_px, refine_owner = [], []
        for did, ((hot_x, hot_y), _) in hotspots.items():
            pts = [(min(max(hot_x + dx, 0), frame_w - 1), min(max(hot_y + dy, 0), frame_h - 1))
                   for dx, dy in [(-r, 0), (r, 0), (0, -r), (0, r)]]
            refine_px.extend(pts)
            refine_owner.extend([did] * len(pts))

        refine_readings = self._query_batch(refine_px, frame_w, frame_h) if refine_px else []

        refine_by_det = {}
        for owner, t in zip(refine_owner, refine_readings):
            if t is not None:
                refine_by_det.setdefault(owner, []).append(t)

        temps = {}
        for d in dets:
            did = id(d)
            if did not in hotspots:
                continue
            (_, hot_t) = hotspots[did]
            chosen = [hot_t] + refine_by_det.get(did, [])
            temps[did] = sum(chosen) / len(chosen)
        return temps

    def _cache_temp(self, cls, xn, yn, temp, now):
        with self._lock:
            self._temp_cache.insert(0, {"cls": cls, "xn": xn, "yn": yn, "temp": temp, "ts": now})
            del self._temp_cache[20:]  # bounded; stale entries also expire by age below

    def _lookup_cached_temp(self, cls, xn, yn, now):
        """Nearest same-species cache entry within match distance + hold window."""
        with self._lock:
            entries = list(self._temp_cache)
        best = None
        best_dist = TEMP_CACHE_MATCH_DIST
        for e in entries:
            if e["cls"] != cls or (now - e["ts"]) > self.temp_hold_seconds:
                continue
            dist = ((e["xn"] - xn) ** 2 + (e["yn"] - yn) ** 2) ** 0.5
            if dist <= best_dist:
                best = e["temp"]
                best_dist = dist
        return best

    def status(self):
        return {
            "fever_margin": self.fever_margin,
            "query_interval": self.query_interval,
            "max_queries_per_cycle": self.max_queries_per_cycle,
            "temp_hold_seconds": self.temp_hold_seconds,
            "known_species": sorted(self.temp_ranges.keys()),
        }
