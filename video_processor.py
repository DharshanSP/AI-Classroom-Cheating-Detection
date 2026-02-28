import cv2
import os
from detection.phone_detector import detect_phone
from detection.head_pose import detect_head_turn
from detection.talking_detector import detect_talking


def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"status": "Failed to open video"}

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    cheating_frames = []
    cheating_score = 0

    previous_center = None
    head_turn_counter = 0

    previous_mouth_area = None
    talk_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analyze every 15th frame for performance
        if frame_count % 15 == 0:

            # =========================
            # PHONE DETECTION
            # =========================
            phone_detected, phone_boxes = detect_phone(frame)

            if phone_detected:
                cheating_score += 5

                timestamp_sec = frame_count / fps
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"

                for (x1, y1, x2, y2) in phone_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                snapshot_path = f"static/snapshots/phone_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)

                cheating_frames.append({
                    "type": "Mobile Phone Detected",
                    "timestamp": timestamp,
                    "snapshot": snapshot_path
                })

            # =========================
            # HEAD TURN DETECTION
            # =========================
            head_turn_detected, previous_center, movement = detect_head_turn(
                frame, previous_center
            )

            if head_turn_detected:
                head_turn_counter += 1
            else:
                head_turn_counter = 0

            if head_turn_counter >= 3:
                cheating_score += 2

                timestamp_sec = frame_count / fps
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"

                snapshot_path = f"static/snapshots/head_turn_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)

                cheating_frames.append({
                    "type": "Repeated Head Turning (Possible Copying)",
                    "timestamp": timestamp,
                    "snapshot": snapshot_path
                })

                head_turn_counter = 0

            # =========================
            # TALKING DETECTION
            # =========================
            talking_detected, previous_mouth_area = detect_talking(
                frame, previous_mouth_area
            )

            if talking_detected:
                talk_counter += 1
            else:
                talk_counter = 0

            if talk_counter >= 3:
                cheating_score += 3

                timestamp_sec = frame_count / fps
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"

                snapshot_path = f"static/snapshots/talking_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)

                cheating_frames.append({
                    "type": "Talking Detected (Possible Cheating)",
                    "timestamp": timestamp,
                    "snapshot": snapshot_path
                })

                talk_counter = 0

    cap.release()

    # =========================
    # FINAL DECISION
    # =========================
    if cheating_score == 0:
        final_status = "No Cheating Detected"
    elif cheating_score < 5:
        final_status = "Suspicious Behavior"
    else:
        final_status = "Cheating Likely"

    return {
        "status": final_status,
        "total_frames": frame_count,
        "cheating_score": cheating_score,
        "events": cheating_frames
    }