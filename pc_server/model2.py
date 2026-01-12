import cv2
from inference import get_model
from dotenv import load_dotenv
import os

load_dotenv()

# --- KONFIGURACJA ---
API_KEY = os.getenv("API_KEY")
KEYPOINTS_MODEL = "cue-detection-ciazj/3"

def main():
    # Pobieramy model
    model_kp = get_model(model_id=KEYPOINTS_MODEL, api_key=API_KEY)
    cap = cv2.VideoCapture(0)

    print("Model: Cue detection (Only keypoints). Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        annotated_frame = frame.copy()

        # 1. Wnioskowanie
        results_kp = model_kp.infer(frame)[0]

        # 2. Rysowanie tylko kropek w miejscach wykrytych punktów
        if hasattr(results_kp, "predictions"):
            for pred in results_kp.predictions:
                if hasattr(pred, "keypoints"):
                    for kp in pred.keypoints:
                        # Rysujemy tylko, jeśli pewność > 40%
                        if kp.confidence > 0.4:
                            x, y = int(kp.x), int(kp.y)
                            
                            if kp.class_name == "tip":
                                # Czerwona kropka (TIP)
                                cv2.circle(annotated_frame, (x, y), 6, (0, 0, 255), -1)
                                
                            elif kp.class_name == "handle" or kp.class_name == "grip":
                                # Niebieska kropka (HANDLE)
                                cv2.circle(annotated_frame, (x, y), 6, (255, 0, 0), -1)

        cv2.imshow("Cue Detection (Keypoints Only)", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()