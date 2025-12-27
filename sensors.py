import cv2
import numpy as np
import math
import socket
import json
import threading
import time
import csv
import os
from datetime import datetime
from collections import deque
import itertools

# --- CONFIGURATION ---
PORT = 5555
HISTORY_SIZE = 200
CONFIG_FILE = "config.json"

# Files and Folders
CSV_FILE = "hit_history.csv"
IMAGE_FOLDER = "saved_graphs"

# Delays
DELAY_SNAPSHOT_1 = 0.8  
DELAY_SNAPSHOT_2 = 3.0  

# --- DATA FROM PLAGENHOEF ET AL. (1983) ---
# Sum of Hand + Forearm + Upper Arm percentages
# Source: Table 4, Segment Weights as Percentages of Total Body Weight
ARM_PERCENTAGE_MALE = 0.0577    # 0.65% + 1.87% + 3.25%
ARM_PERCENTAGE_FEMALE = 0.0497  # 0.5% + 1.57% + 2.9%

def load_arm_mass():
    """
    Calculates arm mass based on Plagenhoef et al. (1983) data.
    """
    default_mass = 4.0
    
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️ Config file not found. Using default arm mass: {default_mass} kg")
        return default_mass

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        weight = float(config.get("body_weight_kg", 75.0))
        gender = config.get("gender", "male").lower()
        glove = float(config.get("glove_weight_kg", 0.0)) # Extra weight for equipment
        
        # Calculate biological arm mass based on research percentages
        if gender == "female":
            bio_arm_mass = weight * ARM_PERCENTAGE_FEMALE
            print(f"ℹ️ Loaded settings for FEMALE: {weight}kg -> Arm mass: {bio_arm_mass:.2f} kg (4.97%)")
        else:
            bio_arm_mass = weight * ARM_PERCENTAGE_MALE
            print(f"ℹ️ Loaded settings for MALE: {weight}kg -> Arm mass: {bio_arm_mass:.2f} kg (5.77%)")
            
        total_mass = bio_arm_mass + glove
        print(f"✅ Total Effective Mass (Bio + Glove): {total_mass:.2f} kg")
        return total_mass

    except Exception as e:
        print(f"❌ Error reading config: {e}. Using default: {default_mass} kg")
        return default_mass

# PHYSICAL CONSTANTS - CALCULATED DYNAMICALLY
ESTIMATED_ARM_MASS = load_arm_mass()

# Thresholds
THRESHOLD_HIT = 20.0  # m/s^2 (Hit detection threshold)
THRESHOLD_PEAK = 15.0 # m/s^2 (Threshold to recognize a peak in acceleration)

# --- GLOBAL VARIABLES ---
sensor_data = {
    "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
    "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
    "last_update": 0
}

# Session Peaks
peak_acc_current = 0.0    # m/s^2
peak_force_n_current = 0.0 # N (Newtons)
peak_gyro_current = 0.0   # deg/s

# History Deques
history_acc = deque([0]*HISTORY_SIZE, maxlen=HISTORY_SIZE)    # Acceleration
history_force_n = deque([0]*HISTORY_SIZE, maxlen=HISTORY_SIZE) # Force (N)
history_gyro = deque([0]*HISTORY_SIZE, maxlen=HISTORY_SIZE)   # Rotation

shot_trigger_time = 0
waiting_for_snapshot_1 = False
waiting_for_snapshot_2 = False

# Variables to save (Values from impact moment)
saved_acc = 0.0
saved_force_n = 0.0
saved_gyro = 0.0

# Function to save file, updated with Newtons
def save_file(val_acc, val_force_n, val_gyro, image_snapshot):
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # Save only one type of image (MOMENT)
    png_filename = f"{IMAGE_FOLDER}/hit_{timestamp}_MOMENT.png"
    
    # Background for text
    cv2.rectangle(image_snapshot, (0, 550), (800, 600), (0, 0, 0), -1)
    
    desc_mode = "(at impact)"

    # INFO ON IMAGE (Added Newtons)
    info_text = f"ACC: {val_acc:.1f} m/s2 | FORCE: {val_force_n:.0f} N | ROTATION: {val_gyro:.0f} deg/s"
    cv2.putText(image_snapshot, info_text, (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(image_snapshot, f"HIT ANALYSIS {desc_mode}", (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(png_filename, image_snapshot)
    
    # SAVE TO CSV (Added Force_N column)
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["Date", "Time", "Acc_ms2", "Force_N", "Rotation_deg_s", "File"])
        
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        writer.writerow([date_str, time_str, f"{val_acc:.2f}", f"{val_force_n:.1f}", f"{val_gyro:.2f}", png_filename])
        
    print(f"💾 SAVED: Acc: {val_acc:.1f}, Force: {val_force_n:.0f} N, Gyro: {val_gyro:.0f}")

def tcp_server_thread():
    global sensor_data
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    last_packet_time = time.time()

    try:
        server_socket.bind(('0.0.0.0', PORT))
        server_socket.listen(1)
        print(f"📡 Server listening on port {PORT}...")
        
        while True:
            conn, addr = server_socket.accept()
            with conn:
                buffer = ""
                last_packet_time = time.time()

                while True:
                    try:
                        data_chunk = conn.recv(4096)
                        if not data_chunk: break
                        buffer += data_chunk.decode('utf-8')
                        if len(buffer) > 8192: buffer = buffer[-4096:]

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if not line.strip(): continue
                            try:
                                js = json.loads(line)
                                ax, ay, az = float(js.get("acc_x", 0)), float(js.get("acc_y", 0)), float(js.get("acc_z", 0))
                                gx, gy, gz = float(js.get("gyro_x", 0)), float(js.get("gyro_y", 0)), float(js.get("gyro_z", 0))

                                gx, gy, gz = gx * 57.2958, gy * 57.2958, gz * 57.2958

                                now = time.time()
                                last_packet_time = now

                                # 1. Total Acceleration (a)
                                total_acc = math.sqrt(ax**2 + ay**2 + az**2)
                                
                                # 2. Force (F = m*a)
                                force_n = total_acc * ESTIMATED_ARM_MASS
                                
                                # 3. Total Rotation
                                total_gyro = math.sqrt(gx**2 + gy**2 + gz**2)

                                sensor_data["acc_x"] = ax; sensor_data["acc_y"] = ay; sensor_data["acc_z"] = az
                                sensor_data["gyro_x"] = gx; sensor_data["gyro_y"] = gy; sensor_data["gyro_z"] = gz
                                sensor_data["last_update"] = now

                                history_acc.append(total_acc)
                                history_force_n.append(force_n) # Add to history
                                history_gyro.append(total_gyro)

                            except json.JSONDecodeError: pass
                    except Exception: break
    except Exception as e:
        print(f"Server error: {e}")

def draw_graph(img, data, x_start, y_start, width, height, color, scale_factor=2.0, title="", unit="", grid_step=10):
    cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (30, 30, 30), -1)
    cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (100, 100, 100), 1)
    base_line_y = y_start + height - 10
    
    max_val_visible = int(height / scale_factor)
    for val in range(0, max_val_visible, grid_step):
        y_pos = base_line_y - int(val * scale_factor)
        if y_pos < y_start: break
        if val > 0:
            cv2.line(img, (x_start, y_pos), (x_start + width, y_pos), (60, 60, 60), 1)
        cv2.putText(img, str(val), (x_start + 5, y_pos - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    points = []
    if len(data) > 0:
        step = width / len(data)
        for i, value in enumerate(data):
            x = x_start + int(i * step)
            y = base_line_y - int(value * scale_factor)
            y = max(y_start, min(y, base_line_y))
            points.append((x, y))
    
    if len(points) > 1:
        cv2.polylines(img, [np.array(points)], isClosed=False, color=color, thickness=2)

    cv2.putText(img, title, (x_start + 5, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    if data:
        cv2.putText(img, f"{data[-1]:.1f} {unit}", (x_start + width - 120, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    global shot_trigger_time, waiting_for_snapshot_1, waiting_for_snapshot_2
    global saved_acc, saved_force_n, saved_gyro
    global peak_acc_current, peak_force_n_current, peak_gyro_current

    threading.Thread(target=tcp_server_thread, daemon=True).start()

    cv2.namedWindow("Hit Analysis")
    cv2.moveWindow("Hit Analysis", 400, 50)
    print("System ready.")

    while True:
        window = np.zeros((600, 800, 3), dtype=np.uint8)
        current_time = time.time()
        is_connected = (current_time - sensor_data["last_update"]) < 1.0
        
        if is_connected:
            # HUD
            cv2.rectangle(window, (10, 10), (240, 590), (20, 20, 20), -1)
            cv2.rectangle(window, (10, 10), (240, 590), (100, 100, 100), 1)
            cv2.circle(window, (30, 40), 8, (0, 255, 0), -1) 
            cv2.putText(window, "ONLINE", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if waiting_for_snapshot_1:
                curr_acc = history_acc[-1] if history_acc else 0
                curr_force_n = history_force_n[-1] if history_force_n else 0
                curr_gyro = history_gyro[-1] if history_gyro else 0
                
                if curr_acc > peak_acc_current: peak_acc_current = curr_acc
                if curr_force_n > peak_force_n_current: peak_force_n_current = curr_force_n
                if curr_gyro > peak_gyro_current: peak_gyro_current = curr_gyro

            # HUD Display (Session Peak Values)
            disp_acc = peak_acc_current if waiting_for_snapshot_1 else saved_acc
            disp_force_n = peak_force_n_current if waiting_for_snapshot_1 else saved_force_n
            disp_gyro = peak_gyro_current if waiting_for_snapshot_1 else saved_gyro

            # 1. ACCELERATION
            cv2.putText(window, "MAX ACC:", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(window, f"{disp_acc:.1f}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(window, "m/s^2", (160, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # 2. FORCE (NEWTONS) - NEW
            cv2.putText(window, "MAX FORCE (F=ma):", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(window, f"{disp_force_n:.0f}", (25, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3) # Orange color
            cv2.putText(window, "N", (160, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

            # 3. ROTATION
            cv2.putText(window, "MAX ROTATION:", (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(window, f"{disp_gyro:.0f}", (25, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
            cv2.putText(window, "deg/s", (160, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

            # GRAPHS (Now 3 pieces)
            # 1. ACCELERATION (Top, Red)
            draw_graph(window, list(history_acc), 260, 20, 520, 160, (0, 0, 255), 
                       scale_factor=3.0, title="ACCELERATION (Acc)", unit="m/s2", grid_step=10)

            # 2. FORCE N (Middle, Orange) - Smaller scale because values are large (e.g. 40g * 4kg = 160N)
            draw_graph(window, list(history_force_n), 260, 200, 520, 160, (0, 165, 255), 
                       scale_factor=0.8, title="HIT FORCE (F=ma)", unit="N", grid_step=50)
            
            # 3. ROTATION (Bottom, Purple)
            draw_graph(window, list(history_gyro), 260, 380, 520, 160, (255, 0, 255), 
                       scale_factor=0.3, title="ROTATION (Gyro)", unit="deg/s", grid_step=100)

            # --- HIT DETECTION ---
            acc_now = history_acc[-1] if history_acc else 0

            if acc_now > THRESHOLD_HIT and not waiting_for_snapshot_1 and not waiting_for_snapshot_2:
                shot_trigger_time = current_time
                # Reset peaks at start of new session
                peak_acc_current = acc_now 
                peak_force_n_current = acc_now * ESTIMATED_ARM_MASS
                peak_gyro_current = 0.0
                
                waiting_for_snapshot_1 = True
                waiting_for_snapshot_2 = True

            # Snapshot 1 (MOMENT OF IMPACT)
            if waiting_for_snapshot_1 and (current_time - shot_trigger_time > DELAY_SNAPSHOT_1):
                
                hist_a = list(history_acc)
                hist_f = list(history_force_n)
                hist_g = list(history_gyro)

                if hist_a:
                    # INTELLIGENT SEARCH FOR 2ND PEAK (Skipping swing)
                    peaks_indices = []
                    for i in range(1, len(hist_a) - 1):
                        if hist_a[i] > THRESHOLD_PEAK:
                            if hist_a[i] >= hist_a[i-1] and hist_a[i] >= hist_a[i+1]:
                                peaks_indices.append(i)
                    
                    impact_idx = 0
                    if peaks_indices:
                        if len(peaks_indices) > 1:
                            candidates = peaks_indices[1:] # Discard 1st peak (swing)
                            impact_idx = max(candidates, key=lambda i: hist_a[i]) # Highest of the rest
                        else:
                            impact_idx = peaks_indices[0]
                    else:
                        impact_idx = np.argmax(hist_a)

                    # --- GET VALUES FROM IMPACT POINT ---
                    saved_acc = hist_a[impact_idx]
                    
                    if impact_idx < len(hist_f):
                        saved_force_n = hist_f[impact_idx]
                    else:
                        saved_force_n = peak_force_n_current

                    if impact_idx < len(hist_g):
                        saved_gyro = hist_g[impact_idx]
                    else:
                        saved_gyro = peak_gyro_current
                    
                    # Draw Line
                    graph_width = 520
                    x_start = 260
                    step = graph_width / len(hist_a)
                    impact_x = x_start + int(impact_idx * step)
                    cv2.line(window, (impact_x, 20), (impact_x, 540), (255, 255, 255), 1)
                    cv2.putText(window, "IMPACT", (impact_x + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                snapshot = window.copy()
                # Saving only this one image
                threading.Thread(target=save_file, args=(saved_acc, saved_force_n, saved_gyro, snapshot), daemon=True).start()
                waiting_for_snapshot_1 = False

            # End of session (without saving second image)
            if waiting_for_snapshot_2 and (current_time - shot_trigger_time > DELAY_SNAPSHOT_2):
                waiting_for_snapshot_2 = False

            if waiting_for_snapshot_1:
                cv2.putText(window, "ANALYZING...", (30, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        else:
            cv2.putText(window, "WAITING...", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Hit Analysis", window)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()