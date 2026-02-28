import cv2
from ultralytics import YOLO

model = YOLO("./yolov8n.pt")

def detect_talking(frame, previous_mouth_area):

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Approximate mouth area (lower 1/3 of face box)
                mouth_y1 = int(y1 + (y2 - y1) * 0.6)
                mouth_region = frame[mouth_y1:y2, x1:x2]

                if mouth_region.size == 0:
                    return False, previous_mouth_area

                gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                mouth_area = cv2.countNonZero(gray)

                if previous_mouth_area is None:
                    return False, mouth_area

                movement = abs(mouth_area - previous_mouth_area)

                if movement > 2000:  # threshold
                    return True, mouth_area

                return False, mouth_area

    return False, previous_mouth_area