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


class ThermalCamClient:
    def __init__(self, base_url: str, timeout: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def video_feed_url(self, color_map: str = None) -> str:
        url = f"{self.base_url}/video_feed"
        if color_map:
            url += f"?map={color_map}"
        return url

    def get_pixel_temp(self, x_norm: float, y_norm: float):
        """Estimated temperature (°C) at a normalized (0-1, 0-1) frame point."""
        try:
            r = requests.get(f"{self.base_url}/pixel_temp",
                              params={"x": x_norm, "y": y_norm}, timeout=self.timeout)
            if r.ok:
                return r.json().get("temp")
        except requests.RequestException:
            pass
        return None

    def get_temperature_data(self):
        """{"ambient","center","status","fps"} or None if unreachable."""
        try:
            r = requests.get(f"{self.base_url}/temperature_data", timeout=self.timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def get_temp_range(self):
        """{"min","max","min_xy","max_xy","color_map","status"} or None."""
        try:
            r = requests.get(f"{self.base_url}/temp_range", timeout=self.timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def get_system_status(self):
        """{"running","mlx_connected","camera_running","frame_count","fps","color_map"} or None."""
        try:
            r = requests.get(f"{self.base_url}/system_status", timeout=self.timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return None
