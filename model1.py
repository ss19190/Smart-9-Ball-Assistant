import cv2
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
import os

load_dotenv()

# --- KONFIGURACJA ---
API_KEY = os.getenv("API_KEY")
MODEL_DETEKCJI = "8-pool-anrdr/2"

def main():
    # Pobieramy model
    model_obj = get_model(model_id=MODEL_DETEKCJI, api_key=API_KEY)

    # Kamera (zostawiłem indeks 1, jak w twoim kodzie)
    cap = cv2.VideoCapture(0)

    # Annotator ustawiony tylko na rysowanie ramki
    box_annotator = sv.BoxAnnotator(thickness=2)

    print("Model: Ball detection.  Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Wnioskowanie
        results_obj = model_obj.infer(frame)[0]
        detections = sv.Detections.from_inference(results_obj)

        # 2. Rysowanie tylko ramek (bez napisów)
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        
        cv2.imshow("Ball Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
