import pickle
import os
import numpy as np
from utils import preprocess_image, get_embedding

embeddings_db = {}
DB_FILE = 'embeddings.pkl'

if os.path.exists(DB_FILE):
    with open(DB_FILE, 'rb') as f:
        embeddings_db = pickle.load(f)

def batch_enroll(uid, folder_path):
    emb_list = []
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.jpg', '.jpeg')):
            path = os.path.join(folder_path, img_file)
            try:
                preprocessed = preprocess_image(path)
                emb = get_embedding(preprocessed)
                emb_list.append(emb)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    if emb_list:
        avg_emb = np.mean(emb_list, axis=0)
        embeddings_db[uid] = avg_emb / np.linalg.norm(avg_emb)
        with open(DB_FILE, 'wb') as f:
            pickle.dump(embeddings_db, f)
        print(f"Enrolled and saved {uid} with {len(emb_list)} images.")
    else:
        print("No valid images found.")

if __name__ == "__main__":
    batch_enroll("STUDENT_23L31A4434", r"data\facebank\STUDENT_23L31A4434")
    batch_enroll("STUDENT_23L31A4464", r"data\facebank\STUDENT_23L31A4464")