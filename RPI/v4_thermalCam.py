from flask import Flask, Response, send_from_directory, jsonify, request
import cv2
import numpy as np
import threading
import time
import requests
import socket
from device_scanner import scan_devices

app = Flask(__name__, static_folder="static")

latest_frame = None
calibrated_point = None  # Holds latest calibrated data from Jetson
coord_buffer = []
buffer_size = 60
send_interval = 2.0
lock = threading.Lock()

# Placeholder IPs
esp32_base = ""
esp32_status = ""
esp32_move = ""
esp32_config = ""
esp32_locked = False
jetson_ip = ""
jetson_endpoint = ""

# Function to implement device_scanner.py
def resolve_device_ips():
    global esp32_base, esp32_status, esp32_move, esp32_config, esp32_locked
    global jetson_ip, jetson_endpoint
    
    if esp32_locked:
        return
    
    devices = scan_devices()
    for d in devices:
        print(d) # debug to find the ip of devices
        ip = d['ip']
        hostname = d['hostname'].lower()
        vendor = d['vendor']

        if 'esp32' in hostname or 'Espressif Inc.' in vendor:
            esp32_base = f"http://{ip}"
            esp32_status = f"{esp32_base}/status"
            esp32_move = f"{esp32_base}/move"
            esp32_config = f"{esp32_base}/config"
            esp32_locked = True
            print(f"IP of esp32 {esp32_base}")
        elif 'nvidia' in vendor.lower() or 'orinnano' in hostname.lower():
            jetson_ip = f"http://{ip}:8081"
            jetson_endpoint = f"http://{ip}:8081/status"

# Function to discover camera attached to raspberry pi
def find_working_camera(max_devices=5):
    print("Searching for camera...")
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at /dev/video{i}")
            cap.release()
            return i
    print("No working camera found")
    return None

# Function to capture thermal feed
def capture_thermal_feed():
    global latest_frame, calibrated_point

    cam_index = find_working_camera()
    if cam_index is None:
        print("Exiting capture thread — no camera found.")
        latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            time.sleep(0.1)
            continue

        # convert to gray image if not received gray image
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # find highest pixel intensity
        _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
        
        # convert gray image to color image
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # labelling max coordinate
        cv2.putText(colored, f"Coords: {maxLoc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored, f"FPS: {1 / (time.time() - start_time):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(colored, maxLoc, 10, (0, 255, 255), 2)
        
        # Pointing max temperature pixel if available
        if calibrated_point:
            # Retrieve scale variable from Jetson
            scale = calibrated_point.get("scale", 0)
            # Max pixel intensity x scale
            est_temp = round(scale * maxVal, 1)
            # Pixel coordinate of max intensity
            position = (maxLoc[0] + 15, maxLoc[1] - 15)
            # Display text of Max Temperature
            cv2.putText(colored, f"Max Temp : {est_temp:.1f}C", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(colored, f"Max Intensity: {maxVal:.1f}", (maxLoc[0] + 15, maxLoc[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Retrieve detected object coordinate from Jetson
        if calibrated_point:
            x, y = calibrated_point.get("x", 0), calibrated_point.get("y", 0)
            
            # retrieve scale variable
            scale = calibrated_point.get("scale",0)
            
            # Displaying object temperature
            if 0 <= x < gray.shape[1] and 0 <= y <gray.shape[0]:
                est_temp = round(float(gray[y,x]) * scale, 1)
                cv2.putText(colored, f"Temp: 30C", (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(colored, (x, y), 8, (0, 255, 0), 2)

        latest_frame = colored.copy()
        frame_count += 1
        if frame_count == 1:
            print("First video frame captured and stored.")

        with lock:
            coord_buffer.append(maxLoc)
            if len(coord_buffer) > buffer_size:
                coord_buffer.pop(0)

        time.sleep(0.01)

# Function to send coordinates to esp32
def send_to_esp32_loop():
    global esp32_move
    print("Waiting for ESP32 move URL...")
    
    # Wait until esp32_move is valid
    while not esp32_move or not esp32_move.startswith("http"):
        time.sleep(0.5)
    
    print(f"ESP32 move URL set: {esp32_move}")
    
    while True:
        time.sleep(send_interval)
        with lock:
            if len(coord_buffer) < buffer_size:
                continue
            avg_x = sum(p[0] for p in coord_buffer) // buffer_size
            avg_y = sum(p[1] for p in coord_buffer) // buffer_size
            averaged = (avg_x, avg_y)

        try:
            response = requests.post(
                esp32_move,
                json={'x': averaged[0], 'y': averaged[1]},
                timeout=0.2
            )
            if response.status_code == 200:
                print(f"Sent average: {averaged}")
            else:
                print(f"ESP32 error code: {response.status_code}")
        except requests.RequestException as e:
            print(f"ESP32 unreachable: {e}")


# http route of ready UI
@app.route('/')
def index():
    return send_from_directory("static", "index.html")

# http route of R&D UI
@app.route('/v3')
def index_v3():
    return send_from_directory("static", "v3_Index.html")

# http route of raw video feed
@app.route('/video_feed')
def video_feed():
    global latest_frame
    retries = 0
    while latest_frame is None and retries < 100:
        print("Waiting for first frame...")
        time.sleep(0.1)
        retries += 1

    if latest_frame is None:
        print("Timeout: using fallback blank frame.")
        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        print("First frame ready — starting stream.")
        fallback = latest_frame

    def generate():
        while True:
            frame = latest_frame if latest_frame is not None else fallback
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# http route of esp32 servo configuration    
@app.route('/proxy_config', methods=['POST'])
def proxy_config():
    try:
        response = requests.post(esp32_config, data=request.form, timeout=1)
        return response.text, response.status_code
    except Exception as e:
        print(f"Proxy failed: {e}")
        return "ESP32 unreachable", 500

# http route of esp32 status
@app.route('/proxy_servo')
def proxy_servo():
    resolve_device_ips()
    try:
        response = requests.get(f"{esp32_base}/servo", timeout=0.5)
        return response.json()
    except:
        return jsonify({"error": "ESP32 unreachable"}), 500

# Function to retrieve rpi ip address
def get_wlan_ip():
    try:
        iface = "wlan0"
        import fcntl, struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,
            struct.pack('256s', iface.encode('utf-8'))
        )[20:24])
    except:
        return "Unavailable"

# route of sidebar status (UI)
@app.route('/status.json')
def status_json():
    resolve_device_ips()

    esp32_alive = False
    jetson_alive = False

    try:
        r = requests.get(esp32_status, timeout=0.5)
        esp32_alive = r.ok
    except:
        pass

    try:
        r = requests.get(jetson_endpoint, timeout=0.5)
        jetson_alive = r.ok
    except:
        pass
    
    scale_value = calibrated_point.get("scale", -1) if calibrated_point else -1
    return jsonify({
        "esp32": esp32_alive,
        "jetson": jetson_alive,
        "ip_esp32": esp32_base,
        "ip_jetson": jetson_ip,
        "ip_server": f"http://{get_wlan_ip()}:8080",
        "scale": round(scale_value, 4) if scale_value > 0 else "-"
    })

# route to retrieve calibration from Jetson
@app.route('/calibrate', methods=['POST'])
def receive_calibration():
    global calibrated_point
    data = request.json
    x = data["x"]
    y = data["y"]
    temp = data["temp"]
    intensity = data["intensity"]
    if intensity == 0:
        return jsonify({"error": "Invalid pixel intensity (0)"}), 400
    scale = temp / intensity
    
    calibrated_point = {
         "x": x,
         "y": y,
         "temp": temp,
         "intensity": intensity,
         "scale": scale
    }
    print(f"Received calibrated point: {calibrated_point}")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    threading.Thread(target=capture_thermal_feed, daemon=True).start()
    threading.Thread(target=send_to_esp32_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)
