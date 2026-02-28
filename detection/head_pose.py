from ultralytics import YOLO

model = YOLO("./yolov8n.pt")

def detect_head_turn(frame, previous_center):

    results = model(frame)

    person_centers = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                person_centers.append(center_x)

    if len(person_centers) == 0:
        return False, previous_center, None

    current_center = person_centers[0]

    if previous_center is None:
        return False, current_center, None

    movement = abs(current_center - previous_center)

    if movement > 40:
        return True, current_center, movement

    return False, current_center, movement