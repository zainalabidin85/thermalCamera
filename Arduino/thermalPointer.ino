// final version known to be working on UI not check with physical movement of servos.

#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <ESPmDNS.h>

const char* ssid = "thermalServer";  // or your AP name
const char* password = "";

WebServer server(80);
Servo servoX, servoY;

// Servo range (configurable via /config)
int servoX_min = 20;
int servoX_max = 160;
int servoY_min = 30;
int servoY_max = 150;

// Global variable for exposing servo angle in UI
int currentX = 90;
int currentY = 90;

// function for servos movement
void handleMove() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "No body received");
    return;
  }

  DynamicJsonDocument doc(512);
  DeserializationError err = deserializeJson(doc, server.arg("plain"));
  if (err) {
    server.send(400, "text/plain", "JSON parse error");
    return;
  }

  int x = doc["x"];
  int y = doc["y"];

  int angleX = constrain(map(x, 0, 640, servoX_min, servoX_max), 0, 180);
  int angleY = constrain(map(y, 0, 480, servoY_min, servoY_max), 0, 180);

  servoX.write(angleX);
  servoY.write(angleY);
  currentX = angleX;
  currentY = angleY;

  Serial.printf("📍 Received x=%d y=%d → angleX=%d angleY=%d\n", x, y, angleX, angleY);
  server.send(200, "text/plain", "Servo moved.");
}

// function for current servos angle
void handleServoStatus(){
  DynamicJsonDocument doc(128);
  doc["x"] = currentX;
  doc["y"] = currentY;

  String json;
  serializeJson(doc, json);
  json += "\n";
  server.send(200, "application/json", json);
}

void handleStatus() {
  server.send(200, "application/json", "{\"status\":\"ok\"}");
}

// function for servo configuration
void handleConfig() {
  if (server.hasArg("xMin")) servoX_min = server.arg("xMin").toInt();
  if (server.hasArg("xMax")) servoX_max = server.arg("xMax").toInt();
  if (server.hasArg("yMin")) servoY_min = server.arg("yMin").toInt();
  if (server.hasArg("yMax")) servoY_max = server.arg("yMax").toInt();

  Serial.printf("🔧 Updated servo limits: X[%d–%d], Y[%d–%d]\n",
                servoX_min, servoX_max, servoY_min, servoY_max);

  server.send(200, "text/plain", "Servo Range Updated");
}

// function for network
void printNetworkInfo() {
  Serial.println("📡 Network Details:");
  Serial.print("IP Address: ");     Serial.println(WiFi.localIP());
  Serial.print("MAC: ");            Serial.println(WiFi.macAddress());
  Serial.print("SSID: ");           Serial.println(WiFi.SSID());
  Serial.print("RSSI: ");           Serial.println(WiFi.RSSI());
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.begin(ssid, password);
  Serial.print("🔌 Connecting to WiFi");
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry++ < 20) {
    Serial.print(".");
    delay(500);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected");
    printNetworkInfo();
  } else {
    Serial.println("\n❌ Failed to connect to WiFi");
    return;
  }

  if (MDNS.begin("esp32")) {
    Serial.println("✅ mDNS responder started");
  } else {
    Serial.println("❌ mDNS setup failed");
  }

  servoX.attach(20);  // use GPIO 20 for X
  servoY.attach(21);  // use GPIO 21 for Y
  servoX.write(currentX);
  servoY.write(currentY);

  server.on("/move", HTTP_POST, handleMove);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/config", HTTP_POST, handleConfig);
  server.on("/servo", handleServoStatus);

  server.begin();
  Serial.println("🌐 Web server started");
}

void loop() {
  server.handleClient();
}
