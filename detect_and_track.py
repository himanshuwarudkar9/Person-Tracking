import os
import cv2
import torch
import numpy as np
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import csv
from reid_utils import ReIDMatcher, trackid_to_globalid  # Import the global mapping

# Configs
VIDEO_DIR = path/of/video/directory
OUTPUT_DIR = 'output'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Models
model = YOLO('yolov8m.pt')  # or yolov8l.pt for better accuracy
tracker = DeepSort(max_age=20, n_init=3)  # Tuned for shops
face_model = YOLO(r'D:\Projects\CV-Project\yolov8n-face.pt')  # Download and use a YOLOv8 face model
matcher = ReIDMatcher(similarity_thresh=0.75, model_name='osnet_x1_0',face_model=face_model) #model_name='osnet_ain_x1_0',similarity_thresh=0.8

# Data storage
csv_rows = [('id', 'video', 'frame', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')]
all_embeddings = []  # For cross-video clustering

def process_video(video_path, video_idx):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video
    out_path = os.path.join(OUTPUT_DIR, video_name.replace('.mp4', '_annotated.mp4'))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people (YOLOv8)
        results = model(frame, conf=0.5, classes=[0])[0]  # Filter for persons + confidence
        boxes = results.boxes
        detections = boxes.xyxy.cpu().numpy() if boxes is not None else []
        confs = boxes.conf.cpu().numpy() if boxes is not None else []

        # Format for DeepSORT
        dets_for_sort = []
        for det, conf in zip(detections, confs):
            x1, y1, x2, y2 = map(int, det[:4])
            w, h = x2 - x1, y2 - y1
            if w * h < 1000:  # Skip small bboxes
                continue
            dets_for_sort.append(([x1, y1, w, h], conf, 'person'))

        # Track with DeepSORT
        tracks = tracker.update_tracks(dets_for_sort, frame=frame)

        # ReID and annotate
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            crop = frame[y1:y2, x1:x2]

            global_id, embedding, face_bbox = matcher.assign_id(crop, video_idx=video_idx)
            if global_id != -1 and face_bbox is not None:
                # Assign global ID to track if not already assigned
                if track.track_id not in trackid_to_globalid:
                    trackid_to_globalid[track.track_id] = global_id
                # Always use the mapped global ID for this track
                global_id = trackid_to_globalid[track.track_id]

                fx, fy, fw, fh = face_bbox
                cv2.rectangle(frame, (x1+fx, y1+fy), (x1+fx+fw, y1+fy+fh), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {global_id}', (x1+fx, y1+fy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                all_embeddings.append((global_id, embedding))
                csv_rows.append((global_id, video_name, frame_idx, x1+fx, y1+fy, fw, fh))
            # else: do nothing if no face

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
# from sklearn.cluster import DBSCAN
# import numpy as np
# from collections import Counter

# def merge_ids_across_videos():
#     global all_embeddings, csv_rows
#     if not all_embeddings:
#         return

#     embeddings = np.array([emb for _, emb in all_embeddings])
#     clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine').fit(embeddings)
#     cluster_labels = clustering.labels_

#     # Build a mapping from old_id to all cluster labels it was assigned
#     id_to_clusters = {}
#     for (old_id, _), cluster_id in zip(all_embeddings, cluster_labels):
#         id_to_clusters.setdefault(old_id, []).append(cluster_id)

#     # For each old_id, assign the most common cluster label
#     id_map = {}
#     for old_id, clusters in id_to_clusters.items():
#         most_common_cluster = Counter(clusters).most_common(1)[0][0]
#         id_map[old_id] = most_common_cluster + 1  # Start IDs from 1

#     # Update csv_rows with merged IDs
#     new_csv_rows = [csv_rows[0]]  # header
#     for row in csv_rows[1:]:
#         old_id = row[0]
#         if old_id in id_map:
#             row = list(row)
#             row[0] = id_map[old_id]
#         new_csv_rows.append(row)
#     csv_rows = new_csv_rows
if __name__ == '__main__':
    video_files = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    for idx, video_file in enumerate(video_files):
        print(f'Processing {video_file}')
        process_video(video_file, video_idx=idx)

    # Merge IDs across videos (optional, for even more robust global IDs)
    #merge_ids_across_videos()

    csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f'Done. Outputs saved in {OUTPUT_DIR}/')
