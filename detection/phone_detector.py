from ultralytics import YOLO

# Load YOLO model once
model = YOLO("./yolov8n.pt")

def detect_phone(frame):
    results = model(frame)

    phone_detected = False
    phone_boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phone_boxes.append((x1, y1, x2, y2))

    return phone_detected, phone_boxes