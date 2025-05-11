import os
import cv2
import torch
import numpy as np
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import csv
from reid_utils import ReIDMatcher
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Configs
VIDEO_DIR = r'path/to/the/Input/Video/Directory'
OUTPUT_DIR = 'output'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Models
model = YOLO('yolov8m.pt')  # or yolov8l.pt for better accuracy
tracker = DeepSort(max_age=20, n_init=3)  # Tuned for shops
matcher = ReIDMatcher(similarity_thresh=0.8, model_name='osnet_ain_x1_0')  # or 'osnet_ibn_x1_0'

# Data storage
csv_rows = [('id', 'video', 'frame', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')]
all_embeddings = []  # For cross-video clustering

def process_video(video_path):
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

            global_id, embedding = matcher.assign_id(crop)
            if global_id != -1:
                all_embeddings.append((global_id, embedding))  # Save for clustering
                csv_rows.append((global_id, video_name, frame_idx, x1, y1, x2-x1, y2-y1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {global_id}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
# ALso checked by applying clusterring methods
# from collections import defaultdict
# from sklearn.cluster import DBSCAN

# def merge_ids_across_videos():
#     """Merge IDs using DBSCAN clustering on mean embeddings per local ID."""
#     if not all_embeddings:
#         return

#     # Group embeddings by local ID
#     id_to_embeds = defaultdict(list)
#     for local_id, emb in all_embeddings:
#         id_to_embeds[local_id].append(emb)

#     # Compute mean embedding for each local ID
#     mean_embeddings = []
#     local_ids = []
#     for local_id, embeds in id_to_embeds.items():
#         mean_emb = np.mean(embeds, axis=0)
#         mean_emb = mean_emb / np.linalg.norm(mean_emb)
#         mean_embeddings.append(mean_emb)
#         local_ids.append(local_id)

#     mean_embeddings = np.array(mean_embeddings)
#     # Cluster
#     clustering = DBSCAN(eps=0.4, min_samples=1).fit(mean_embeddings)
#     # After clustering
#     id_mapping = {local_id: int(cluster_id) + 1 for local_id, cluster_id in zip(local_ids, clustering.labels_)}
#     # Update CSV rows with merged IDs
#     for i, row in enumerate(csv_rows[1:], 1):  # Skip header
#         row = list(row)
#         row[0] = id_mapping.get(row[0], row[0])  # Use cluster ID if available
#         csv_rows[i] = tuple(row)
# def merge_ids_across_videos():
#     """Cluster all embeddings and assign global IDs based on clusters."""
#     if not all_embeddings:
#         return

#     # Unpack all embeddings and metadata
#     meta = []
#     feats = []
#     for video_name, frame_idx, x1, y1, x2, y2, local_id, emb in all_embeddings:
#         meta.append((video_name, frame_idx, x1, y1, x2, y2, local_id))
#         feats.append(emb)
#     feats = np.array(feats)
#     feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

#     # Cluster all embeddings
#     clustering = DBSCAN(eps=0.4, min_samples=1).fit(feats)
#     cluster_labels = clustering.labels_

#     # Build a mapping from (video, frame, bbox, local_id) to cluster label
#     meta_to_cluster = {}
#     for m, cluster_id in zip(meta, cluster_labels):
#         meta_to_cluster[m] = int(cluster_id) + 1  # Start IDs from 1

#     # Update CSV rows with merged IDs
#     for i, row in enumerate(csv_rows[1:], 1):  # Skip header
#         video_name, frame_idx, x1, y1, w, h = row[1:7]
#         x2 = int(x1) + int(w)
#         y2 = int(y1) + int(h)
#         local_id = row[0]
#         key = (video_name, int(frame_idx), int(x1), int(y1), int(x2), int(y2), local_id)
#         new_id = meta_to_cluster.get(key, local_id)
#         row = list(row)
#         row[0] = new_id
#         csv_rows[i] = tuple(row)
if __name__ == '__main__':
    video_files = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    for video_file in video_files:
        print(f'Processing {video_file}')
        process_video(video_file)

    # Merge IDs across videos
    #merge_ids_across_videos()

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f'Done. Outputs saved in {OUTPUT_DIR}/')
