# Smart Pool Assistant & Stroke Analyzer 🎱🤖

A comprehensive billiards training system combining **Computer Vision** (table situation analysis) and **Biomechanics** (stroke analysis using Android sensors).

This project consists of two main modules:
1.  **AI Vision Assistant:** Real-time tracking of balls and cue stick, calculating the "Ghost Ball" position, and visualizing shot trajectories.
2.  **Sensor Stroke Analysis:** Analysis of G-force, impact force ($F=ma$), and hand stability using an Android smartphone attached to the player's arm.

## 🚀 Key Features

### 👁️ Vision Module (Python + OpenCV + Roboflow)
* **Object Detection:** Identifies the Cue Ball and Object Balls using the `8-pool-anrdr` model.
* **Cue Tracking:** Detects Cue Tip & Handle keypoints to determine the aiming vector in real-time.
* **Physics Prediction:**
    * Calculates the **Ghost Ball** position (the point of impact).
    * Visualizes the **Tangent Line** (Cue Ball path) and **Normal Line** (Object Ball path) based on the 90-degree rule.
    * Distinguishes between valid collisions and misses.

### 📉 Sensor Module (Python + Android)
* **Telemetry Transmission:** Android app sends high-frequency **Linear Acceleration** (gravity removed) and **Gyroscope** data via TCP/IP.
* **Stroke Analysis:**
    * **Peak Detection:** Automatically detects the moment of impact.
    * **Force Calculation:** Estimates impact force in Newtons ($F = m \times a$) based on arm mass.
    * **Stability Check:** Monitors wrist rotation (Gyro) to detect unwanted twists during the stroke.
* **Visualization:** Custom real-time graphing engine built purely in OpenCV for high performance.
* **Data Logging:** Saves hit history to CSV and captures snapshots of the impact metrics.

## 🛠️ Tech Stack

**Backend & Vision:**
* **Language:** Python 3.x
* **Libraries:** `opencv-python` (Visualization), `supervision` (Annotation), `inference` (Roboflow API), `numpy` (Vector math).
* **Networking:** Socket & Threading (TCP server).

**Mobile (Android):**
* **Language:** Java.
* **Sensors:** `TYPE_LINEAR_ACCELERATION`, `TYPE_GYROSCOPE`.
* **Comms:** TCP Sockets (Client).

## ⚙️ Installation & Usage

### Prerequisites
Install the required Python libraries:

```bash
pip install -r req.txt
```

### Running the Vision Assistant
1. Connect a webcam pointed at the pool table.
2. Run desktop application

```bash
python bilard.py
```

### Running the Stroke Analyzer
This module requires an Android device connected to the PC through USB cable. 

1. Install the .apk file from this repositorium to your device.

2. Setup ADB Forwarding: This allows the phone to talk to localhost on your PC over USB.

```bash
adb reverse tcp:5555 tcp:5555
```

3. Start the Server:
```bash
python sensors.py
```

4. Open the Android app and tap CONNECT.

## 📂 Project Structure
- bilard.py - Main AR script (Detection, Tracking, Trajectory Prediction).

- sensors.py - TCP Server, Real-time Graphing, Hit Analysis logic.

- model1.py - Unit test: Ball detection only.

- model2.py - Unit test: Cue stick keypoint detection only.

- sensorApp.apk - Android app

## 🧠 How it works (Technical Details)
**Ghost Ball Prediction**: The system projects the cue stick vector onto the vector connecting the cue ball and the target ball. If the perpendicular distance is less than the ball radius, it calculates the "Ghost Ball" center. It then uses vector algebra to draw the predicted paths (tangent/normal vectors).

**Biomechanics Analysis**: 
- **Force** ($F=ma$): The app strips Earth's gravity from the accelerometer readings. The Python server multiplies this raw acceleration by the estimated arm mass (default: 4.0 kg) to derive the Force in Newtons.
- **Rotation** (Technique Check): The Gyroscope data ($x, y, z$) is aggregated into a total magnitude ($\sqrt{x^2+y^2+z^2}$). This metric serves as a "Technique Score" — the lower the rotation during the forward stroke, the more stable and professional the cue delivery is.