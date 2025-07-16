import os
import subprocess
import socket
import requests
import threading

# Known MAC address prefixes for vendor identification
VENDOR_PREFIXES = {
    "DC:06:75": "Espressif Inc.",
    "3C:71:BF": "Espressif Inc.",
    "DC:A6:32": "Raspberry Pi",
    "48:B0:2D": "NVIDIA",
    "00:04:4B": "NVIDIA",
    "00:1B:FC": "NVIDIA",
    "14:13:33": "NVIDIA"  # Jetson Orin Nano
}

def lookup_vendor(mac):
    prefix = mac.upper()[0:8]
    return VENDOR_PREFIXES.get(prefix, "Unknown")

def ping_ip(ip):
    # Ping once to populate ARP cache
    subprocess.run(["ping", "-c", "1", "-W", "1", ip], stdout=subprocess.DEVNULL)

def async_ping_subnet(subnet="10.42.0", max_hosts=254):
    threads = []
    for i in range(1, max_hosts):
        ip = f"{subnet}.{i}"
        t = threading.Thread(target=ping_ip, args=(ip,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def get_active_devices():
    devices = []
    if not os.path.exists('/proc/net/arp'):
        return devices

    with open('/proc/net/arp') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                ip = parts[0]
                mac = parts[3]
                if mac != "00:00:00:00:00:00":
                    try:
                        hostname = socket.gethostbyaddr(ip)[0]
                    except socket.herror:
                        hostname = "Unknown"
                    vendor = lookup_vendor(mac)
                    print(f"Discovered: IP={ip}, MAC={mac}, Host={hostname}, Vendor={vendor}")
                    devices.append({
                        'ip': ip,
                        'mac': mac,
                        'hostname': hostname,
                        'vendor': vendor
                    })
    return devices

def validate_status(ip, port=8081):  # Port 8081 for Jetson Flask
    try:
        r = requests.get(f"http://{ip}:{port}/status", timeout=0.3)
        return r.ok
    except requests.RequestException:
        return False

def scan_devices():
    print("Scanning subnet for ESP32 and Jetson devices...")
    async_ping_subnet()  # Populate ARP table
    all_devices = get_active_devices()
    valid_devices = []

    for d in all_devices:
        if d['vendor'] in ['Espressif Inc.', 'NVIDIA']:
            # ESP32 uses port 80, Jetson uses port 8081
            port = 80 if d['vendor'] == 'Espressif Inc.' else 8081
            if validate_status(d['ip'], port=port):
                print(f"{d['vendor']} at {d['ip']} passed /status check")
                valid_devices.append(d)
            else:
                print(f"{d['vendor']} at {d['ip']} did not respond on /status (port {port})")
    return valid_devices
