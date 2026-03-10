from flask import Flask, request, jsonify
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist
from collections import defaultdict
import numpy as np
import cv2
import time
import logging
import os

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

# -------------------------
# Load YOLO Model
# -------------------------
model = YOLO("nano.pt")

# -------------------------
# Parameters
# -------------------------
DISTANCE_THRESHOLD  = 60      # px — close pair threshold
RISK_THRESHOLD      = 0.65    # risk score to trigger alert
DBSCAN_EPS          = 80      # px — cluster radius
DBSCAN_MIN_SAMPLES  = 2       # min people to form a cluster
DENSITY_RADIUS      = 100     # px — neighbourhood radius for local density
FIXED_SIZE          = (640, 480)
STALE_SESSION_SEC   = 30      # reset motion if frame gap > 30s

# -------------------------
# Per-Camera Session Store
# {camera_id: {"prev_gray": np.array, "prev_centers": list, "timestamp": float}}
# -------------------------
camera_sessions = defaultdict(lambda: {
    "prev_gray":    None,
    "prev_centers": [],
    "timestamp":    0.0
})

# -------------------------
# Crowd Analysis
# -------------------------
def analyze_crowd(frame, camera_id="cam_01"):

    session  = camera_sessions[camera_id]
    now      = time.time()
    h, w     = frame.shape[:2]

    # ── YOLO Head Detection ───────────────────────────────────────
    # Uses Abcfsa/YOLOv8_head_detector — outputs normalized xywh coords
    results  = model(frame)
    centers  = []

    for box in results[0].boxes:
        # xywhn → normalized center_x, center_y, width, height
        cx_n, cy_n, _, _ = box.xywhn[0].tolist()
        cx = int(cx_n * w)   # denormalize to pixel coords
        cy = int(cy_n * h)
        centers.append((cx, cy))

    head_count = len(centers)

    # ── Density (heads per 1000 px²) ─────────────────────────────
    area    = (w * h) / 1000.0
    density = head_count / area if area > 0 else 0.0

    # ── Close-Pair Distance Factor ────────────────────────────────
    avg_distance    = 0.0
    distance_factor = 0.0

    if head_count > 1:
        dists           = pdist(np.array(centers))
        avg_distance    = float(np.mean(dists))
        close_pairs     = np.sum(dists < DISTANCE_THRESHOLD)
        distance_factor = float(close_pairs / len(dists))

    # ── DBSCAN Crowd Clustering ───────────────────────────────────
    n_clusters   = 0
    max_cluster  = 0
    cluster_risk = 0.0

    if head_count >= DBSCAN_MIN_SAMPLES:
        pts    = np.array(centers)
        db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
        labels = db.labels_
        unique = set(labels) - {-1}
        n_clusters = len(unique)

        if n_clusters > 0:
            sizes        = [np.sum(labels == lbl) for lbl in unique]
            max_cluster  = int(max(sizes))
            cluster_risk = float(max_cluster / head_count)

    # ── Local Density via BallTree ────────────────────────────────
    local_density = 0.0

    if head_count > 1:
        pts  = np.array(centers, dtype=np.float64)
        tree = BallTree(pts)
        neighbour_counts = tree.query_radius(pts, r=DENSITY_RADIUS, count_only=True)
        local_density = float(np.mean(neighbour_counts - 1))

    # ── Inter-Frame Motion ────────────────────────────────────────
    # Only computed if previous frame exists and arrived within STALE_SESSION_SEC
    motion   = 0.0
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray     = cv2.resize(gray, FIXED_SIZE)
    prev     = session["prev_gray"]
    prev_ts  = session["timestamp"]
    time_gap = now - prev_ts

    if prev is not None and time_gap < STALE_SESSION_SEC:
        try:
            flow   = cv2.calcOpticalFlowFarneback(
                prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion = float(np.mean(mag))
        except cv2.error as e:
            logger.warning(f"[{camera_id}] Optical flow skipped: {e}")
            motion = 0.0
    else:
        if prev is not None:
            logger.info(f"[{camera_id}] Stale session reset ({time_gap:.1f}s gap)")

    # Update session
    session["prev_gray"]    = gray
    session["prev_centers"] = centers
    session["timestamp"]    = now

    # ── Risk Score (weighted blend, capped at 1.0) ────────────────
    risk_score = (
        density         * 0.30 +
        distance_factor * 0.25 +
        cluster_risk    * 0.25 +
        local_density   * 0.10 +
        motion          * 0.10
    )
    risk_score = float(min(risk_score, 1.0))
    alert      = bool(risk_score > RISK_THRESHOLD)

    # ── Alert Level ───────────────────────────────────────────────
    if risk_score < 0.35:
        level = "LOW"
    elif risk_score < 0.65:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "camera_id":       camera_id,
        "head_count":      head_count,
        "density":         round(density, 4),
        "avg_distance":    round(avg_distance, 2),
        "distance_factor": round(distance_factor, 4),
        "n_clusters":      n_clusters,
        "max_cluster":     max_cluster,
        "cluster_risk":    round(cluster_risk, 4),
        "local_density":   round(local_density, 4),
        "motion":          round(motion, 4),
        "risk_score":      round(risk_score, 4),
        "alert":           alert,
        "level":           level,
        "timestamp":       round(now, 2)
    }

# -------------------------
# POST /analyze
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"error": "image missing"}), 400

    camera_id = request.form.get("camera_id", "cam_01")
    file      = request.files["image"]
    npimg     = np.frombuffer(file.read(), np.uint8)
    frame     = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "could not decode image — check JPEG quality on ESP32"}), 400

    try:
        result = analyze_crowd(frame, camera_id)
        logger.info(
            f"[{camera_id}] heads={result['head_count']} "
            f"risk={result['risk_score']} level={result['level']}"
        )
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[{camera_id}] Analysis failed: {e}", exc_info=True)
        return jsonify({"error": "analysis failed", "detail": str(e)}), 500

# -------------------------
# GET /health
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "nano.pt"}), 200

# -------------------------
# GET /
# -------------------------
@app.route("/")
def home():
    return "Crowd Management API Running", 200


if __name__ == "__main__":
    # Render injects PORT via environment variable — must read it here
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
