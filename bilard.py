import cv2
import supervision as sv
from inference import get_model
import numpy as np
import math
from dotenv import load_dotenv
import os

load_dotenv()

# --- CONFIGURATION ---

API_KEY = os.getenv("API_KEY")  
DETECTION_MODEL = "ball-detection-bzirz/3"
KEYPOINTS_MODEL = "cue-detection-ciazj/3"

# --- IMPORTANT: Adjust this to the ball size on screen ---
BALL_DIAMETER_PX = 45 

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
    
    if perpendicular_dist < BALL_DIAMETER_PX:
        back_offset = math.sqrt(BALL_DIAMETER_PX**2 - perpendicular_dist**2)
        distance_to_impact = projection_length - back_offset
        ghost_ball_pos = cue_ball_pos + aim_vector * distance_to_impact
        return ghost_ball_pos, distance_to_impact
    
    return None, float('inf')

def main():
    model_obj = get_model(model_id=DETECTION_MODEL, api_key=API_KEY)
    model_kp = get_model(model_id=KEYPOINTS_MODEL, api_key=API_KEY)

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Enable if you have an HD camera

    box_annotator = sv.BoxAnnotator(thickness=2)
    # label_annotator = sv.LabelAnnotator() # I disabled labels so they don't obscure the lines

    print("Start! Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        annotated_frame = frame.copy()

        # 1. OBJECT DETECTION
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
            elif class_name == "other": # or other name for colored balls
                other_balls_centers.append(center_point)

        # 2. CUE KEYPOINT DETECTION
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

        # 3. DRAWING LINES AND PREDICTION
        if cue_ball_center is not None and tip_pos is not None and handle_pos is not None:
            
            # --- DRAWING CUE LINE (Handle -> Tip) ---
            handle_int = tuple(handle_pos.astype(int))
            tip_int = tuple(tip_pos.astype(int))
            # Thick line representing the cue
            cv2.line(annotated_frame, handle_int, tip_int, (255, 0, 0), 4) 

            # Calculate aiming vector based on the cue
            aim_vector_raw = tip_pos - handle_pos
            aim_direction = normalize_vector(aim_vector_raw)

            # Looking for collision
            closest_ghost_ball = None
            closest_target_ball = None
            min_distance = float('inf')

            for target_ball in other_balls_centers:
                ghost_pos, distance = find_ghost_ball_position(cue_ball_center, aim_direction, target_ball)
                if ghost_pos is not None and distance < min_distance:
                    min_distance = distance
                    closest_ghost_ball = ghost_pos
                    closest_target_ball = target_ball

            # --- DRAWING PREDICTION LINES ---
            cue_start_int = tuple(cue_ball_center.astype(int))
            
            # Line connecting Tip to Cue Ball (shows "aiming")
            cv2.line(annotated_frame, tip_int, cue_start_int, (100, 100, 100), 2, cv2.LINE_AA)

            if closest_ghost_ball is not None:
                ghost_int = tuple(closest_ghost_ball.astype(int))
                target_int = tuple(closest_target_ball.astype(int))

                # 1. Line Cue Ball -> Ghost Ball (continuation of cue line)
                cv2.line(annotated_frame, cue_start_int, ghost_int, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Circle at ghost ball location
                cv2.circle(annotated_frame, ghost_int, int(BALL_DIAMETER_PX/2), (255, 255, 255), 1)

                # Rebound physics
                impact_vector = closest_target_ball - closest_ghost_ball
                impact_direction = normalize_vector(impact_vector)
                
                tangent_direction = np.array([-impact_direction[1], impact_direction[0]])
                if np.dot(tangent_direction, aim_direction) < 0:
                     tangent_direction = -tangent_direction

                # 2. Target Ball Path (Green)
                target_end_pos = closest_target_ball + impact_direction * 200
                cv2.arrowedLine(annotated_frame, target_int, tuple(target_end_pos.astype(int)), (0, 255, 0), 4, tipLength=0.2)

                # 3. Cue Ball Path (Yellow)
                cue_end_pos = closest_ghost_ball + tangent_direction * 200
                cv2.arrowedLine(annotated_frame, ghost_int, tuple(cue_end_pos.astype(int)), (0, 255, 255), 4, tipLength=0.2)
            else:
                # If no collision, draw line "to infinity" from the cue ball
                infinity_point = cue_ball_center + aim_direction * 1000
                cv2.line(annotated_frame, cue_start_int, tuple(infinity_point.astype(int)), (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Billiards AI - Full Path", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
