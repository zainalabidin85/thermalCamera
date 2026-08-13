// final version known to be working on UI not check with physical movement of servos.
/*
  Thermal Pointer Control Firmware for ESP32
  ------------------------------------------
  Receives coordinates via HTTP POST from Flask.
  Converts to servo angles and adjusts orientation.

  WiFi setup: no SSID/password is hardcoded. On first boot (or after a
  credential reset), the ESP32 opens its own AP named "ThermalPointer-Setup"
  serving a captive-portal page — connect to it from a phone/laptop and
  it walks you through picking your WiFi network and entering the
  password. WiFiManager stores the result in flash (NVS) and reconnects
  to it automatically on every subsequent boot; the portal only reappears
  if the saved network can't be reached, or after POST /wifi_reset.
*/


#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <ESPmDNS.h>
#include <WiFiManager.h>

WebServer server(80);
WiFiManager wifiManager;
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

  Serial.printf("Received x=%d y=%d → angleX=%d angleY=%d\n", x, y, angleX, angleY);
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

// clears saved WiFi credentials and reboots into the setup portal
void handleWifiReset() {
  server.send(200, "text/plain", "WiFi credentials cleared. Rebooting into setup portal...");
  delay(200);
  wifiManager.resetSettings();
  ESP.restart();
}

// function for servo configuration
void handleConfig() {
  if (server.hasArg("xMin")) servoX_min = server.arg("xMin").toInt();
  if (server.hasArg("xMax")) servoX_max = server.arg("xMax").toInt();
  if (server.hasArg("yMin")) servoY_min = server.arg("yMin").toInt();
  if (server.hasArg("yMax")) servoY_max = server.arg("yMax").toInt();

  Serial.printf("Updated servo limits: X[%d–%d], Y[%d–%d]\n",
                servoX_min, servoX_max, servoY_min, servoY_max);

  server.send(200, "text/plain", "Servo Range Updated");
}

// function for network
void printNetworkInfo() {
  Serial.println("Network Details:");
  Serial.print("IP Address: ");     Serial.println(WiFi.localIP());
  Serial.print("MAC: ");            Serial.println(WiFi.macAddress());
  Serial.print("SSID: ");           Serial.println(WiFi.SSID());
  Serial.print("RSSI: ");           Serial.println(WiFi.RSSI());
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Blocks until connected to a saved network, or until the user finishes
  // the captive-portal setup flow on the "ThermalPointer-Setup" AP.
  wifiManager.setConfigPortalTimeout(180);  // give up and reboot after 3 min unconfigured
  if (!wifiManager.autoConnect("ThermalPointer-Setup")) {
    Serial.println("\n Failed to connect / configure WiFi within timeout — restarting");
    delay(1000);
    ESP.restart();
  }

  Serial.println("\n WiFi connected");
  printNetworkInfo();

  if (MDNS.begin("esp32")) {
    Serial.println(" mDNS responder started");
  } else {
    Serial.println(" mDNS setup failed");
  }

  servoX.attach(25);  // use GPIO 25 for X
  servoY.attach(26);  // use GPIO 26 for Y
  servoX.write(currentX);
  servoY.write(currentY);

  server.on("/move", HTTP_POST, handleMove);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/config", HTTP_POST, handleConfig);
  server.on("/servo", handleServoStatus);
  server.on("/wifi_reset", HTTP_POST, handleWifiReset);

  server.begin();
  Serial.println(" Web server started");
}

void loop() {
  server.handleClient();
}
