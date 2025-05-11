# 🧠 Multi-Video Person Tracking & Re-Identification

This project performs **multi-video person tracking and ReID (Re-identification)** using YOLOv8 for detection and OSNet for identity matching. It assigns consistent IDs to individuals across different surveillance videos of a shop environment.

---

## 🎯 Project Goal

- Detect people accurately in each video
- Track them over time using Deep SORT
- Assign the **same ID to the same person across different videos**
- Export annotated videos and detection logs
- Achieve high accuracy with **limited compute**

---

## 📦 Pipeline Overview

1. **Detection**: YOLOv8m detects people in each frame.
2. **Tracking**: Deep SORT maintains tracking within a video.
3. **Re-Identification**: OSNet ReID model extracts embeddings to match people across videos.
4. **Global Identity Matching**: Assigns consistent IDs based on cosine similarity between embeddings.
5. **Annotation**: Output videos are saved with bounding boxes and IDs.
6. **Logging**: Results are saved as a CSV.

---
## ⚙️ Installation

### pip install -r requirements.txt
### Ensure you have Python ≥ 3.8 and a compatible CUDA setup if using GPU

## 🚀 Running the Code
Prepare videos, Put all your .mp4 video files in the VIDEO_DIR.

1. Run the main script:- python detect_and_track.py

2. Outputs:- Annotated videos in /output -results.csv file with the following format: (id, video, frame, bbox_x, bbox_y, bbox_w, bbox_h)

##🧪 Models Tested
We tested multiple models for performance vs accuracy:

##🔍 ReID Models (OSNet variants)
osnet_x1_0, osnet_x0_75, osnet_ibn_x1_0, osnet_ain_x1_0, etc.

##🧍 YOLOv8 Variants
yolov8n, yolov8s, yolov8m

##✅ Final choice:
YOLOv8m + OSNet_AIN_x1_0

Balanced speed and accuracy; achieved 80%+ global identity consistency.

## 📈 Evaluation Criteria
Criteria	Achieved
✔ Person detection accuracy	Used YOLOv8m with class=person filtering
✔ Cross-video identity consistency	Global ReID via OSNet with cosine similarity
✔ Code quality and modularity	Modular files: detect_and_track.py, reid_utils.py
✔ Annotated visual results	Saved annotated .mp4 videos with IDs

📁 Repository Structure
📦 root/
 ┣ 📜 detect_and_track.py
 ┣ 📜 reid_utils.py
 ┣ 📁 output/
 ┃ ┗ 📜 results.csv, *_annotated.mp4
 ┗ 📁 videos/
    ┗ 📜 video1.mp4, video2.mp4, ...

📌 Notes
Small bounding boxes (< 1000 px²) are ignored to reduce noise.

Embedding similarity threshold: 0.8

Stored up to 30 embeddings per person to avoid identity drift.
