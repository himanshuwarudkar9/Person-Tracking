import torch
import torchreid
import cv2
import numpy as np
from collections import defaultdict
import insightface

# Persistent mapping for all videos
trackid_to_globalid = {}

class ReIDMatcher:
    def __init__(self, similarity_thresh=0.75, max_embeds=30, model_name='osnet_x1_0', face_model=None):
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.similarity_thresh = similarity_thresh
        self.known_embeddings = defaultdict(list)
        self.next_global_id = 1
        self.max_embeds = max_embeds
        self.face_model = face_model

        # Load InsightFace for face embeddings
        self.face_embedder = insightface.app.FaceAnalysis(name='buffalo_l')
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.face_embedder.prepare(ctx_id=ctx_id)

    def preprocess(self, img):
        if img is None or img.size == 0 or min(img.shape[:2]) < 20:
            return None
        img = cv2.resize(img, (256, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img = (img - mean) / std
        if torch.cuda.is_available():
            img = img.cuda()
        return img

    def extract_features(self, img):
        img = self.preprocess(img)
        if img is None:
            return None
        with torch.no_grad():
            features = self.model(img)
        features = features.cpu().numpy().flatten()
        features /= np.linalg.norm(features)
        return features

    def detect_face(self, img):
        if self.face_model is None:
            return []
        results = self.face_model(img, conf=0.5)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            faces.append([x1, y1, x2 - x1, y2 - y1])
        return faces

    def extract_face_features(self, face_crop):
        # face_crop: BGR image (OpenCV)
        faces = self.face_embedder.get(face_crop)
        if not faces:
            return None
        # Use the largest detected face in the crop
        face = max(faces, key=lambda f: f.bbox[2]*f.bbox[3])
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
        return emb

    def assign_id(self, img, video_idx=0):
        if img is None or img.size == 0:
            return -1, None, None

        faces = self.detect_face(img)
        face_emb = None
        face_bbox = None
        if len(faces) > 0:
            face_bbox = max(faces, key=lambda b: b[2]*b[3])
            x, y, w, h = face_bbox
            face_crop = img[y:y+h, x:x+w]
            face_emb = self.extract_face_features(face_crop)

        body_emb = self.extract_features(img)

        # Determine embedding sizes
        body_dim = 512 if body_emb is not None else 0
        face_dim = 512 if face_emb is not None else 0  # InsightFace is usually 512

        # Pad missing parts with zeros so all embeddings are 1024-dim
        if body_emb is None:
            body_emb = np.zeros(512)
        if face_emb is None:
            face_emb = np.zeros(512)

        combined_emb = np.concatenate([body_emb, face_emb])
        combined_emb = combined_emb / np.linalg.norm(combined_emb)

        max_sim = -1
        best_id = -1
        for id_, embeddings in self.known_embeddings.items():
            for emb in embeddings:
                emb = emb / np.linalg.norm(emb)
                sim = np.dot(combined_emb, emb)
                if sim > max_sim:
                    max_sim = sim
                    best_id = id_

        if max_sim >= self.similarity_thresh:
            self.known_embeddings[best_id].append(combined_emb)
            if len(self.known_embeddings[best_id]) > self.max_embeds:
                self.known_embeddings[best_id].pop(0)
            return best_id, combined_emb, face_bbox
        else:
            new_id = self.next_global_id
            self.known_embeddings[new_id].append(combined_emb)
            self.next_global_id += 1
            return new_id, combined_emb, face_bbox
