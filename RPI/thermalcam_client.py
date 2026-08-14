#!/usr/bin/env python3
"""thermalcam_client.py - HTTP client for the thermalCam-Pi device.

thermalCam-Pi (a separate project: a USB camera + MLX90614 IR point sensor
on its own Raspberry Pi, default port 5000) is the only device with the
sensor attached. It already does temperature estimation server-side — see
its /pixel_temp endpoint, which linearly extrapolates a temperature at any
normalized (x,y) point in the frame from the MLX90614's center-point
reading, computed against the true raw sensor frame (not a compressed/
color-mapped copy). This client just calls that API; it does not duplicate
any calibration math locally.
"""

import requests
from requests.adapters import HTTPAdapter


class ThermalCamClient:
    def __init__(self, base_url: str, timeout: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Reused across calls (incl. concurrent ones from a thread pool - the
        # underlying HTTPAdapter's connection pool is thread-safe) so bursts
        # of /pixel_temp queries reuse TCP/keep-alive connections instead of
        # paying a fresh handshake per call - thermalCam-Pi is a Raspberry Pi
        # already busy with its own camera loop, so every bit of per-request
        # overhead matters. Pool bumped past requests' default of 10 since a
        # single hotspot-search batch can fire more concurrent calls than that.
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=32)
        self._session.mount(self.base_url, adapter)

    def video_feed_url(self, color_map: str = None) -> str:
        url = f"{self.base_url}/video_feed"
        if color_map:
            url += f"?map={color_map}"
        return url

    def get_pixel_temp(self, x_norm: float, y_norm: float):
        """Estimated temperature (°C) at a normalized (0-1, 0-1) frame point."""
        try:
            r = self._session.get(f"{self.base_url}/pixel_temp",
                              params={"x": x_norm, "y": y_norm}, timeout=self.timeout)
            if r.ok:
                return r.json().get("temp")
        except requests.RequestException:
            pass
        return None

    def get_temperature_data(self):
        """{"ambient","center","status","fps"} or None if unreachable."""
        try:
            r = self._session.get(f"{self.base_url}/temperature_data", timeout=self.timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def get_temp_range(self):
        """{"min","max","min_xy","max_xy","color_map","status"} or None."""
        try:
            r = self._session.get(f"{self.base_url}/temp_range", timeout=self.timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def get_system_status(self):
        """{"running","mlx_connected","camera_running","frame_count","fps","color_map"} or None."""
        try:
            r = self._session.get(f"{self.base_url}/system_status", timeout=self.timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return None
