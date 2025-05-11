# import torch
# import torchreid
# import cv2
# import numpy as np

# class ReIDMatcher:
#     def __init__(self, similarity_thresh=0.7):
#         # Initialize model
#         self.model = torchreid.models.build_model(
#             name='osnet_ain_x1_0',
#             num_classes=1000,
#             pretrained=True
#         )
#         self.model.eval()
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()
            
#         self.similarity_thresh = similarity_thresh
#         self.known_embeddings = {}
#         self.next_id = 0
        
#     def extract_features(self, img):
#         # Preprocess image
#         img = cv2.resize(img, (256, 128))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = torch.from_numpy(img).float()
#         img = img.permute(2, 0, 1)  # Change to CHW format
#         img = img.unsqueeze(0)  # Add batch dimension
#         img = img / 255.0  # Normalize to [0,1]
        
#         if torch.cuda.is_available():
#             img = img.cuda()
            
#         # Extract features
#         with torch.no_grad():
#             features = self.model(img)
            
#         return features.cpu().numpy()
    
#     def compute_similarity(self, feat1, feat2):
#         return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
#     def assign_id(self, img):
#         if img is None or img.size == 0:
#             return -1
            
#         # Extract features for current detection
#         curr_features = self.extract_features(img)
        
#         # If no known embeddings, assign new ID
#         if not self.known_embeddings:
#             self.known_embeddings[self.next_id] = curr_features
#             self.next_id += 1
#             return self.next_id - 1
            
#         # Compare with known embeddings
#         max_sim = -1
#         best_id = -1
        
#         for id_, feat in self.known_embeddings.items():
#             sim = self.compute_similarity(curr_features.flatten(), feat.flatten())
#             if sim > max_sim:
#                 max_sim = sim
#                 best_id = id_
                
#         # If similarity is high enough, assign existing ID
#         if max_sim >= self.similarity_thresh:
#             return best_id
            
#         # Otherwise, assign new ID
#         self.known_embeddings[self.next_id] = curr_features
#         self.next_id += 1
#         return self.next_id - 1

# import torch
# import torchreid
# import cv2
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# class ReIDMatcher:
#     def __init__(self, similarity_thresh=0.5):
#         # Use stronger model
#         self.model = torchreid.models.build_model(
#             name='osnet_ain_x1_0',  # Better variant
#             num_classes=1000,
#             pretrained=True
#         )
#         self.model.eval()
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()

#         self.similarity_thresh = similarity_thresh
#         self.known_embeddings = {}  # ID -> list of features
#         self.next_id = 0

#     def extract_features(self, img):
#         try:
#             img = cv2.resize(img, (256, 128))
#         except Exception:
#             return None

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = torch.from_numpy(img).float()
#         img = img.permute(2, 0, 1).unsqueeze(0) / 255.0

#         if torch.cuda.is_available():
#             img = img.cuda()

#         with torch.no_grad():
#             features = self.model(img)

#         return features.cpu().numpy()

#     def compute_similarity(self, feat1, feat2):
#         return cosine_similarity([feat1], [feat2])[0][0]

#     def assign_id(self, img):
#         # Sanity check: skip tiny or empty crops
#         if img is None or img.size == 0 or img.shape[0] < 20 or img.shape[1] < 20:
#             return -1

#         curr_features = self.extract_features(img)
#         if curr_features is None:
#             return -1

#         curr_features = curr_features.flatten()

#         # First ID if no embeddings yet
#         if not self.known_embeddings:
#             self.known_embeddings[self.next_id] = [curr_features]
#             self.next_id += 1
#             return self.next_id - 1

#         max_sim = -1
#         best_id = -1

#         # Compare to all known IDs using average of embeddings
#         for id_, feats in self.known_embeddings.items():
#             mean_feat = np.mean(feats, axis=0)
#             sim = self.compute_similarity(curr_features, mean_feat)
#             if sim > max_sim:
#                 max_sim = sim
#                 best_id = id_

#         if max_sim >= self.similarity_thresh:
#             # Append feature to existing ID
#             self.known_embeddings[best_id].append(curr_features)
#             return best_id

#         # Otherwise assign new ID
#         self.known_embeddings[self.next_id] = [curr_features]
#         self.next_id += 1
#         return self.next_id - 1


# import torch
# import torchreid
# import cv2
# import numpy as np
# from collections import defaultdict

# class ReIDMatcher:
#     def __init__(self, similarity_thresh=0.5, max_embeds=10):
#         self.model = torchreid.models.build_model(
#             name='strong_baseline_t', num_classes=1000, pretrained=True    #osnet_ibn_x1_0 , osnet_ain_x1_0'
#         ).eval()
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()
#         self.similarity_thresh = similarity_thresh
#         self.known_embeddings = {}  # global_id -> list of embeddings
#         self.next_global_id = 0
#         self.max_embeds = max_embeds

#     def preprocess(self, img):
#         if img is None or img.size == 0:
#             return None
#         img = cv2.resize(img, (256, 128))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
#         if torch.cuda.is_available():
#             img = img.cuda()
#         return img

#     def extract_features(self, img):
#         img = self.preprocess(img)
#         if img is None:
#             return None
#         with torch.no_grad():
#             features = self.model(img)
#         return features.cpu().numpy().flatten()

#     def assign_id(self, img):
#         if img is None or img.size == 0 or img.shape[0] < 20 or img.shape[1] < 20:
#             return -1
#         curr_features = self.extract_features(img)
#         if curr_features is None:
#             return -1

#         # If no known embeddings, assign new ID
#         if not self.known_embeddings:
#             self.known_embeddings[self.next_global_id] = [curr_features]
#             self.next_global_id += 1
#             return self.next_global_id - 1

#         max_sim = -1
#         best_id = -1

#         # Compare to all known IDs using max similarity over all stored embeddings
#         for id_, feats in self.known_embeddings.items():
#             for feat in feats:
#                 sim = np.dot(curr_features, feat) / (np.linalg.norm(curr_features) * np.linalg.norm(feat))
#                 if sim > max_sim:
#                     max_sim = sim
#                     best_id = id_

#         if max_sim >= self.similarity_thresh:
#             # Add this embedding to the matched ID, keep only last N
#             self.known_embeddings[best_id].append(curr_features)
#             if len(self.known_embeddings[best_id]) > self.max_embeds:
#                 self.known_embeddings[best_id] = self.known_embeddings[best_id][-self.max_embeds:]
#             return best_id

#         # Otherwise assign new ID
#         self.known_embeddings[self.next_global_id] = [curr_features]
#         self.next_global_id += 1
#         return self.next_global_id - 1


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
