#!/usr/bin/env python3
"""pointer_mapper.py - turns a detection's pixel coordinate into a correct
ESP32 /move payload.

The ESP32 firmware (esp32/thermalPointer/thermalPointer.ino) hard-codes its
pixel space to 640x480 and maps that internally to servo angles via its own
/config-provided xMin/xMax/yMin/yMax range. So the one thing the Pi must get
right is: always report x,y in that same fixed 640x480 reference frame, no
matter what resolution the camera actually captured at. Previous versions
forwarded raw detection-frame pixels straight through, which drifted the
laser off-target whenever DETECT_IMGSZ or the camera's native resolution
didn't match 640x480.
"""

import time
from collections import deque
from threading import Lock

REFERENCE_W = 640
REFERENCE_H = 480


class PointerMapper:
    """Normalizes detection coordinates and rate-limits ESP32 move commands."""

    def __init__(self, move_cooldown=2.0, queue_maxlen=100,
                 reference_w=REFERENCE_W, reference_h=REFERENCE_H):
        self.move_cooldown = float(move_cooldown)
        self.reference_w = int(reference_w)
        self.reference_h = int(reference_h)

        self._lock = Lock()
        self._queue = deque(maxlen=queue_maxlen)
        self._last_move_ts = 0.0

    def normalize(self, cx: int, cy: int, frame_w: int, frame_h: int):
        """Map a pixel coord from the actual frame size into the 640x480
        reference space the ESP32 firmware expects."""
        if frame_w <= 0 or frame_h <= 0:
            return cx, cy
        nx = int(round(cx * (self.reference_w / float(frame_w))))
        ny = int(round(cy * (self.reference_h / float(frame_h))))
        nx = max(0, min(self.reference_w, nx))
        ny = max(0, min(self.reference_h, ny))
        return nx, ny

    def queue_move(self, cx: int, cy: int, frame_w: int, frame_h: int,
                    temp_c: float = None, species: str = None, force=False):
        """Rate-limited: only queues a move if the cooldown has elapsed
        (unless force=True, e.g. a manual UI click)."""
        now = time.time()
        with self._lock:
            if not force and (now - self._last_move_ts) < self.move_cooldown:
                return False
            nx, ny = self.normalize(cx, cy, frame_w, frame_h)
            payload = {"x": nx, "y": ny}
            if temp_c is not None:
                payload["temp"] = round(float(temp_c), 1)
            if species is not None:
                payload["species"] = species
            self._queue.append(payload)
            self._last_move_ts = now
            return True

    def pop(self):
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def requeue_front(self, payload):
        with self._lock:
            self._queue.appendleft(payload)

    def __len__(self):
        with self._lock:
            return len(self._queue)
