# src/main.py
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import time
from collections import deque, defaultdict
import numpy as np

st.set_page_config(page_title="Vehicle Counter Dashboard", layout="wide")
st.title("🚗 Vehicle Detection & Counting Dashboard (Improved Counting)")

with st.sidebar:
    st.header("📊 Live Vehicle Stats")
    bar_chart_placeholder = st.empty()
    line_chart_placeholder = st.empty()

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    st.video(temp_video.name)

    run_button = st.button("▶ Start Processing")

    if run_button:
        st.info("Processing… Please wait.")

        # ---------- Config ----------
        CONF_THR = 0.35       # min confidence to consider detection
        MIN_AREA = 2000       # min bbox area (tune per video)
        SMOOTH_N = 5          # how many centroid Y samples to average (deque length)
        VEL_THR = 2.0         # average dy threshold (pixels/frame) to consider valid motion
        model = YOLO("yolov8n.pt")
        model.overrides['verbose'] = False  # disable console logs

        # Map COCO class ids to names we want to count (keep same mapping)
        vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        cap = cv2.VideoCapture(temp_video.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # counting line (horizontal)
        line_y = int(h * 0.60)

        # trackers for counting
        prev_centroids = defaultdict(lambda: deque(maxlen=SMOOTH_N))  # id -> deque([cy,...])
        counted_ids = set()     # set of ids already counted (one-time)
        class_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

        timeline_df = pd.DataFrame({"time": [], "total": []})
        frame_display = st.empty()

        # output writer
        output_path = os.path.join(tempfile.gettempdir(), "output_improved.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        start_time = time.time()
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # Run detection + tracking - no verbose to avoid terminal spam
            results = model.track(frame, persist=True, verbose=False)[0]

            # draw counting line (red)
            cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 3)

            # process results
            if results.boxes is not None:
                for box in results.boxes:
                    # confidence filter (some boxes may have low conf)
                    conf = float(box.conf[0])
                    if conf < CONF_THR:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_AREA:
                        continue

                    cls_id = int(box.cls[0])
                    if cls_id not in vehicle_classes:
                        continue
                    cls_name = vehicle_classes[cls_id]

                    # tracked ID (from model.track)
                    track_id = int(box.id[0]) if box.id is not None else None
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # update deque for smoothing
                    if track_id is not None:
                        dq = prev_centroids[track_id]
                        dq.append(cy)

                        # only attempt count when we have at least 2 samples
                        if len(dq) >= 2 and track_id not in counted_ids:
                            # compute average velocity over deque (dy per frame)
                            arr = np.array(dq)
                            diffs = np.diff(arr).astype(float)
                            avg_dy = diffs.mean() if len(diffs) > 0 else 0.0

                            # crossing detection: previous mean below line and current mean >= line (downwards)
                            prev_mean = arr.mean() - diffs.mean() if len(diffs) > 0 else arr.mean()
                            curr_mean = arr.mean()

                            crossed_down = (prev_mean < line_y <= curr_mean)
                            crossed_up = (prev_mean > line_y >= curr_mean)

                            # only accept crossing if velocity magnitude is big enough to avoid jitter
                            if (crossed_down or crossed_up) and abs(avg_dy) >= VEL_THR:
                                # count once, optional direction use
                                counted_ids.add(track_id)
                                class_counts[cls_name] += 1

                    # draw bbox, label, red centroid
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID: {track_id} {cls_name}"
                    cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # draw class counts on top-left
            offset = 30
            for k, v in class_counts.items():
                cv2.putText(frame, f"{k}: {v}", (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                offset += 35

            # write frame and show in streamlit
            writer.write(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(rgb)

            # update charts
            bar_df = pd.DataFrame({"Vehicle": list(class_counts.keys()), "Count": list(class_counts.values())})
            bar_chart_placeholder.bar_chart(bar_df, x="Vehicle", y="Count")

            timeline_df.loc[len(timeline_df)] = {"time": round(time.time() - start_time, 1), "total": sum(class_counts.values())}
            line_chart_placeholder.line_chart(timeline_df, x="time", y="total")

        cap.release()
        writer.release()

        st.success("🎉 Processing Completed!")
        st.video(output_path)
        st.write("Saved output:", output_path)

# Sample image path you uploaded (for reference)
st.caption("Sample image you showed: /mnt/data/Output_Sample_image.png")

