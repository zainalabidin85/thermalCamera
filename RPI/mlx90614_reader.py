#!/usr/bin/env python3
# mlx90614_reader.py - OOP MLX90614 temperature reader

import time
import statistics
from collections import deque
from threading import Lock, Thread, Event
import os

# ---------------- MLX90614 backends ----------------
_HAS_ADAFRUIT = False
_HAS_SMBUS2 = False

try:
    import board
    import busio
    from adafruit_mlx90614 import MLX90614
    _HAS_ADAFRUIT = True
except ImportError:
    pass

try:
    from smbus2 import SMBus
    _HAS_SMBUS2 = True
except ImportError:
    pass

# MLX90614 constants
MLX_ADDR = int(os.getenv("MLX90614_ADDR", "0x5A"), 16)
MLX_OFFSET_C = float(os.getenv("MLX90614_OFFSET_C", "0.0"))
REG_TA, REG_TOBJ1 = 0x06, 0x07


class MLXReader:
    """
    MLX90614 infrared temperature sensor reader with multiple backends.
    
    Features:
    - Supports Adafruit and smbus2 backends
    - Moving average filtering
    - Outlier detection and removal
    - Thread-safe operations
    - Background polling
    """
    
    def __init__(self, window_size=5, poll_interval=0.5, i2c_bus=1):
        """
        Initialize MLX90614 reader.
        
        Args:
            window_size (int): Number of samples for moving average
            poll_interval (float): Time between readings in seconds
            i2c_bus (int): I2C bus number (default 1 for Raspberry Pi)
        """
        self.window_size = max(1, int(window_size))
        self.poll_interval = max(0.1, float(poll_interval))
        self.i2c_bus = int(i2c_bus)
        
        # Data storage
        self._ambient_history = deque(maxlen=self.window_size)
        self._object_history = deque(maxlen=self.window_size)
        
        # Hardware state
        self.backend = None
        self._mlx = None
        self._bus = None
        self.ok = False
        
        # Threading
        self._lock = Lock()
        self._stop_event = Event()
        self._thread = None
        
        # Statistics
        self.read_count = 0
        self.error_count = 0
        
        # Initialize hardware
        self._init_hardware()
    
    def _init_hardware(self):
        """Initialize the appropriate hardware backend."""
        if _HAS_ADAFRUIT:
            try:
                self._mlx = MLX90614(busio.I2C(board.SCL, board.SDA))
                self.backend = "adafruit"
                print(f"[MLX90614] Using Adafruit backend")
                return
            except Exception as e:
                print(f"[MLX90614] Adafruit backend failed: {e}")
        
        if _HAS_SMBUS2:
            try:
                self._bus = SMBus(self.i2c_bus)
                self.backend = "smbus2"
                print(f"[MLX90614] Using smbus2 backend on bus {self.i2c_bus}")
                return
            except Exception as e:
                print(f"[MLX90614] smbus2 backend failed: {e}")
        
        self.backend = None
        print("[MLX90614] No working backend available")
    
    def _read_adafruit(self):
        """Read temperature using Adafruit backend."""
        ambient = float(self._mlx.ambient_temperature) + MLX_OFFSET_C
        object_temp = float(self._mlx.object_temperature) + MLX_OFFSET_C
        return ambient, object_temp
    
    def _read_smbus2(self):
        """Read temperature using smbus2 backend."""
        # Read ambient temperature
        raw_ambient = self._bus.read_word_data(MLX_ADDR, REG_TA)
        ambient = raw_ambient * 0.02 - 273.15 + MLX_OFFSET_C
        
        # Read object temperature
        raw_object = self._bus.read_word_data(MLX_ADDR, REG_TOBJ1)
        object_temp = raw_object * 0.02 - 273.15 + MLX_OFFSET_C
        
        return ambient, object_temp
    
    def _read_once(self):
        """Perform a single temperature reading."""
        if self.backend == "adafruit":
            return self._read_adafruit()
        elif self.backend == "smbus2":
            return self._read_smbus2()
        else:
            raise RuntimeError("No MLX90614 backend available")
    
    def read_once(self):
        """
        Perform a single temperature reading and update history.
        
        Returns:
            tuple: (ambient_temp, object_temp) in Celsius
        """
        try:
            ambient, object_temp = self._read_once()
            with self._lock:
                self._ambient_history.append(ambient)
                self._object_history.append(object_temp)
                self.ok = True
                self.read_count += 1
            return ambient, object_temp
        except Exception as e:
            self.error_count += 1
            self.ok = False
            raise e
    
    def _filter_outliers(self, data):
        """
        Remove outliers from data using statistical filtering.
        
        Args:
            data (list): List of temperature values
            
        Returns:
            list: Filtered data without outliers
        """
        if len(data) < 3:
            return data
        
        try:
            mean_val = statistics.mean(data)
            stdev_val = statistics.stdev(data)
            
            # Filter values within 3 standard deviations
            filtered = [x for x in data if abs(x - mean_val) <= 3 * stdev_val]
            return filtered if filtered else data
        except statistics.StatisticsError:
            return data
    
    def get_latest(self, fallback_read=True):
        """
        Get the latest filtered temperature readings.
        
        Args:
            fallback_read (bool): If True, attempt a new reading if no data available
            
        Returns:
            tuple: (ambient_temp, object_temp, status_ok)
        """
        with self._lock:
            has_data = len(self._object_history) > 0
            
            # If no data and fallback enabled, try to read once
            if not has_data and fallback_read and self.backend:
                try:
                    self.read_once()
                    has_data = True
                except Exception:
                    pass
            
            if not has_data:
                return None, None, False
            
            # Apply outlier filtering
            filtered_ambient = self._filter_outliers(list(self._ambient_history))
            filtered_object = self._filter_outliers(list(self._object_history))
            
            # Calculate averages
            ambient_avg = statistics.mean(filtered_ambient) if filtered_ambient else None
            object_avg = statistics.mean(filtered_object) if filtered_object else None
            
            return ambient_avg, object_avg, self.ok
    
    def get_raw_history(self):
        """
        Get raw temperature history for debugging.
        
        Returns:
            tuple: (ambient_history, object_history)
        """
        with self._lock:
            return list(self._ambient_history), list(self._object_history)
    
    def get_stats(self):
        """
        Get reading statistics.
        
        Returns:
            dict: Statistics including read count, error count, etc.
        """
        with self._lock:
            ambient_hist = list(self._ambient_history)
            object_hist = list(self._object_history)
            
            stats = {
                'read_count': self.read_count,
                'error_count': self.error_count,
                'success_rate': self.read_count / (self.read_count + self.error_count) if (self.read_count + self.error_count) > 0 else 0,
                'window_size': len(ambient_hist),
                'backend': self.backend,
                'status': 'OK' if self.ok else 'ERROR'
            }
            
            if ambient_hist:
                stats['ambient_stats'] = {
                    'current': ambient_hist[-1],
                    'min': min(ambient_hist),
                    'max': max(ambient_hist),
                    'avg': statistics.mean(ambient_hist)
                }
            
            if object_hist:
                stats['object_stats'] = {
                    'current': object_hist[-1],
                    'min': min(object_hist),
                    'max': max(object_hist),
                    'avg': statistics.mean(object_hist)
                }
            
            return stats
    
    def _polling_loop(self):
        """Background polling loop for continuous temperature reading."""
        # Initial readings to fill buffer
        for _ in range(min(2, self.window_size)):
            try:
                self.read_once()
            except Exception:
                pass
            time.sleep(self.poll_interval)
        
        # Continuous polling
        while not self._stop_event.is_set():
            try:
                self.read_once()
            except Exception as e:
                print(f"[MLX90614] Polling error: {e}")
            
            time.sleep(self.poll_interval)
    
    def start(self):
        """Start background polling thread."""
        if self._thread and self._thread.is_alive():
            print("[MLX90614] Reader already running")
            return
        
        if not self.backend:
            print("[MLX90614] Cannot start - no backend available")
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._polling_loop, daemon=True)
        self._thread.start()
        print(f"[MLX90614] Started polling every {self.poll_interval}s")
    
    def stop(self):
        """Stop background polling and cleanup resources."""
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self._bus:
            self._bus.close()
            print("[MLX90614] smbus2 connection closed")
        
        print("[MLX90614] Reader stopped")
    
    def is_ready(self):
        """Check if reader has sufficient data."""
        with self._lock:
            return len(self._object_history) >= min(2, self.window_size)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Factory function for easy creation
def create_mlx_reader(window_size=5, poll_interval=0.5, i2c_bus=1):
    """
    Factory function to create and start an MLXReader instance.
    
    Args:
        window_size (int): Number of samples for moving average
        poll_interval (float): Time between readings in seconds
        i2c_bus (int): I2C bus number
        
    Returns:
        MLXReader: Configured and started MLXReader instance
    """
    reader = MLXReader(
        window_size=window_size,
        poll_interval=poll_interval,
        i2c_bus=i2c_bus
    )
    reader.start()
    return reader


if __name__ == "__main__":
    # Test the MLX90614 reader
    print("Testing MLX90614 Reader...")
    
    try:
        with create_mlx_reader(window_size=3, poll_interval=1.0) as reader:
            time.sleep(3)  # Allow some readings to accumulate
            
            for i in range(5):
                ambient, object_temp, ok = reader.get_latest()
                stats = reader.get_stats()
                
                print(f"Reading {i+1}:")
                print(f"  Ambient: {ambient:.2f}°C")
                print(f"  Object: {object_temp:.2f}°C")
                print(f"  Status: {'OK' if ok else 'ERROR'}")
                print(f"  Stats: {stats['read_count']} reads, {stats['error_count']} errors")
                print()
                
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test failed: {e}")
