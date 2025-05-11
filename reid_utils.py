import torch
import torchreid
import cv2
import numpy as np
from collections import defaultdict

class ReIDMatcher:
    def __init__(self, similarity_thresh=0.7, max_embeds=30, model_name='osnet_ain_x1_0'): #checked with 20 earlier
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Feature storage
        self.similarity_thresh = similarity_thresh
        self.known_embeddings = defaultdict(list)  # {id: [embedding1, ...]}
        self.next_global_id = 1
        self.max_embeds = max_embeds

    def preprocess(self, img):
        """Resize and normalize image for ReID model."""
        if img is None or img.size == 0 or min(img.shape[:2]) < 20:
            return None
        
        img = cv2.resize(img, (256, 128))  # OSNet default input size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Normalize (if your model requires it)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img = (img - mean) / std
        
        if torch.cuda.is_available():
            img = img.cuda()
        return img

    def extract_features(self, img):
        """Extract ReID features (L2-normalized)."""
        img = self.preprocess(img)
        if img is None:
            return None
        
        with torch.no_grad():
            features = self.model(img)
        
        features = features.cpu().numpy().flatten()
        features /= np.linalg.norm(features)  # L2 normalize
        return features

    def assign_id(self, img):
        if img is None or img.size < 1000:
            return -1, None

        curr_features = self.extract_features(img)
        if curr_features is None:
            return -1, None

        curr_features = curr_features / np.linalg.norm(curr_features)

        max_sim = -1
        best_id = -1

        for id_, embeddings in self.known_embeddings.items():
            # Compare to all embeddings for this ID
            for emb in embeddings:
                emb = emb / np.linalg.norm(emb)
                sim = np.dot(curr_features, emb)
                if sim > max_sim:
                    max_sim = sim
                    best_id = id_

        #print(f"Best similarity: {max_sim:.3f} for ID {best_id}")  # <-- Add here

        if max_sim >= self.similarity_thresh:
            self.known_embeddings[best_id].append(curr_features)
            if len(self.known_embeddings[best_id]) > self.max_embeds:
                self.known_embeddings[best_id].pop(0)
            return best_id, curr_features
        else:
            new_id = self.next_global_id
            self.known_embeddings[new_id].append(curr_features)
            self.next_global_id += 1
            return new_id, curr_features
