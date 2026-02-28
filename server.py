from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
import math

app = Flask(__name__)

# 🔥 Use better model for crowds (recommended)
model = YOLO("yolov8l.pt")    # change to yolov8l.pt for even better accuracy

previous_centers = {}   # now dictionary: {track_id: (x, y)}
previous_count = 0

RISK_THRESHOLD = 0.3
MAX_EXPECTED_PEOPLE = 36   # adjust based on your scene


def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


@app.route('/process', methods=['POST'])
def process_frame():

    global previous_centers, previous_count

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    height, width = frame.shape[:2]
    frame_area = height * width

    # 🔥 TRACK instead of simple detect
    results = model.track(
        frame,
        persist=True,
        conf=0.02,
        imgsz=1600,
        classes=[0],      # only persons
        iou=0.8,
        max_det=1000
    )

    centers = {}
    count = 0

    for r in results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes
        ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers[track_id] = (cx, cy)
            count += 1

    # -------------------
    # 1️⃣ Density (Stable Normalization)
    # -------------------
    normalized_density = min(count / MAX_EXPECTED_PEOPLE, 1.0)

    # -------------------
    # 2️⃣ Average Distance
    # -------------------
    avg_distance = 0

    if len(centers) > 1:
        coords = list(centers.values())
        distances = []

        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                distances.append(calculate_distance(coords[i], coords[j]))

        avg_distance = np.mean(distances)
    else:
        avg_distance = 999

    inv_distance = 1 / avg_distance if avg_distance != 0 else 1
    normalized_distance = min(inv_distance * 100, 1.0)

    # -------------------
    # 3️⃣ Motion Spike (Correct Tracking-Based)
    # -------------------
    motion_spike = 0

    if previous_centers:
        movements = []

        for track_id, center in centers.items():
            if track_id in previous_centers:
                prev_center = previous_centers[track_id]
                movements.append(
                    calculate_distance(center, prev_center)
                )

        if movements:
            motion_spike = np.mean(movements)

    normalized_motion = min(motion_spike / 50, 1.0)

    # -------------------
    # 4️⃣ Sudden Density Increase
    # -------------------
    density_change = max(count - previous_count, 0)
    normalized_density_change = min(density_change / 20, 1.0)

    # -------------------
    # 5️⃣ Final Risk Score
    # -------------------
    risk = (
        0.35 * normalized_density +
        0.25 * normalized_distance +
        0.25 * normalized_motion +
        0.15 * normalized_density_change
    )

    alert = risk > RISK_THRESHOLD

    previous_centers = centers
    previous_count = count

    return jsonify({
        "people_count": int(count),
        "density": float(normalized_density),
        "avg_distance": float(avg_distance),
        "motion": float(normalized_motion),
        "risk_score": float(risk),
        "ALERT": bool(alert)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
