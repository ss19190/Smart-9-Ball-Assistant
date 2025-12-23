import cv2
import supervision as sv
from inference import get_model
import numpy as np
import math
from dotenv import load_dotenv
import os

load_dotenv()

# --- KONFIGURACJA ---

API_KEY = os.getenv("API_KEY")  
MODEL_DETEKCJI = "8-pool-anrdr/2"
MODEL_KEYPOINTS = "cue-detection/1"

# --- WAŻNE: Dostosuj to do wielkości bili na ekranie ---
SREDNICA_BILI_PX = 45 

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def find_ghost_ball_position(cue_ball_pos, aim_vector, target_ball_pos):
    vec_to_target = target_ball_pos - cue_ball_pos
    projection_length = np.dot(vec_to_target, aim_vector)
    
    if projection_length <= 0: return None, float('inf')

    closest_point_on_line = cue_ball_pos + aim_vector * projection_length
    perpendicular_dist = np.linalg.norm(target_ball_pos - closest_point_on_line)
    
    if perpendicular_dist < SREDNICA_BILI_PX:
        back_offset = math.sqrt(SREDNICA_BILI_PX**2 - perpendicular_dist**2)
        distance_to_impact = projection_length - back_offset
        ghost_ball_pos = cue_ball_pos + aim_vector * distance_to_impact
        return ghost_ball_pos, distance_to_impact
    
    return None, float('inf')

def main():
    model_obj = get_model(model_id=MODEL_DETEKCJI, api_key=API_KEY)
    model_kp = get_model(model_id=MODEL_KEYPOINTS, api_key=API_KEY)

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Włącz jeśli masz kamerę HD

    box_annotator = sv.BoxAnnotator(thickness=2)
    # label_annotator = sv.LabelAnnotator() # Wyłączyłem napisy, żeby nie zasłaniały linii

    print("Start! Naciśnij 'q' aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        annotated_frame = frame.copy()

        # 1. DETEKCJA OBIEKTÓW
        results_obj = model_obj.infer(frame)[0]
        detections = sv.Detections.from_inference(results_obj)

        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        
        cue_ball_center = None
        other_balls_centers = []

        for i in range(len(detections)):
            class_name = detections.data['class_name'][i]
            box = detections.xyxy[i]
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            center_point = np.array([center_x, center_y], dtype=np.float32)

            if class_name == "cue ball" or class_name == "cue-ball":
                cue_ball_center = center_point
            elif class_name == "other": # lub inna nazwa kolorowych bil
                other_balls_centers.append(center_point)

        # 2. DETEKCJA KEYPOINTÓW KIJA
        results_kp = model_kp.infer(frame)[0]
        tip_pos = None
        handle_pos = None

        if hasattr(results_kp, "predictions"):
            for pred in results_kp.predictions:
                if hasattr(pred, "keypoints"):
                    for kp in pred.keypoints:
                        if kp.confidence > 0.4:
                            pos = np.array([kp.x, kp.y], dtype=np.float32)
                            if kp.class_name == "tip":
                                tip_pos = pos
                                cv2.circle(annotated_frame, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), -1)
                            elif kp.class_name == "handle" or kp.class_name == "grip":
                                handle_pos = pos
                                cv2.circle(annotated_frame, (int(pos[0]), int(pos[1])), 5, (255, 0, 0), -1)

        # 3. RYSOWANIE LINII I PREDYKCJA
        if cue_ball_center is not None and tip_pos is not None and handle_pos is not None:
            
            # --- RYSOWANIE LINII KIJA (Handle -> Tip) ---
            handle_int = tuple(handle_pos.astype(int))
            tip_int = tuple(tip_pos.astype(int))
            # Gruba linia odwzorowująca kij
            cv2.line(annotated_frame, handle_int, tip_int, (255, 0, 0), 4) 

            # Obliczamy wektor celowania na podstawie kija
            aim_vector_raw = tip_pos - handle_pos
            aim_direction = normalize_vector(aim_vector_raw)

            # Szukamy kolizji
            closest_ghost_ball = None
            closest_target_ball = None
            min_distance = float('inf')

            for target_ball in other_balls_centers:
                ghost_pos, distance = find_ghost_ball_position(cue_ball_center, aim_direction, target_ball)
                if ghost_pos is not None and distance < min_distance:
                    min_distance = distance
                    closest_ghost_ball = ghost_pos
                    closest_target_ball = target_ball

            # --- RYSOWANIE LINII PREDYKCJI ---
            cue_start_int = tuple(cue_ball_center.astype(int))
            
            # Linia łącząca Tip z Białą Bilą (pokazuje "celowanie")
            cv2.line(annotated_frame, tip_int, cue_start_int, (100, 100, 100), 2, cv2.LINE_AA)

            if closest_ghost_ball is not None:
                ghost_int = tuple(closest_ghost_ball.astype(int))
                target_int = tuple(closest_target_ball.astype(int))

                # 1. Linia Biała -> Duch (ciąg dalszy linii od kija)
                cv2.line(annotated_frame, cue_start_int, ghost_int, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Kółko w miejscu ducha
                cv2.circle(annotated_frame, ghost_int, int(SREDNICA_BILI_PX/2), (255, 255, 255), 1)

                # Fizyka odbicia
                impact_vector = closest_target_ball - closest_ghost_ball
                impact_direction = normalize_vector(impact_vector)
                
                tangent_direction = np.array([-impact_direction[1], impact_direction[0]])
                if np.dot(tangent_direction, aim_direction) < 0:
                     tangent_direction = -tangent_direction

                # 2. Tor Bili Kolorowej (Zielony)
                target_end_pos = closest_target_ball + impact_direction * 200
                cv2.arrowedLine(annotated_frame, target_int, tuple(target_end_pos.astype(int)), (0, 255, 0), 4, tipLength=0.2)

                # 3. Tor Białej Bili (Żółty)
                cue_end_pos = closest_ghost_ball + tangent_direction * 200
                cv2.arrowedLine(annotated_frame, ghost_int, tuple(cue_end_pos.astype(int)), (0, 255, 255), 4, tipLength=0.2)
            else:
                # Jeśli nie ma kolizji, rysujemy linię "w nieskończoność" od białej bili
                infinity_point = cue_ball_center + aim_direction * 1000
                cv2.line(annotated_frame, cue_start_int, tuple(infinity_point.astype(int)), (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Bilard AI - Full Path", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
