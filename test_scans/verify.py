import pickle
import numpy as np
from utils import preprocess_image, get_embedding, compare_embeddings

def load_db():
    with open('embeddings.pkl', 'rb') as f:
        return pickle.load(f)

def verify(image_path, threshold=0.6):
    db = load_db()
    try:
        preprocessed = preprocess_image(image_path)
        scan_emb = get_embedding(preprocessed)
        similarities = {uid: compare_embeddings(scan_emb, emb) for uid, emb in db.items()}
        best_uid = max(similarities, key=similarities.get)
        if similarities[best_uid] > threshold:
            return best_uid
    except Exception as e:
        print(f"Error verifying {image_path}: {e}")
    return None

if __name__ == "__main__":
    test_image = "E:\python_proto\data\facebank\STUDENT_23L31A4434"  # Change to your path
    matched_uid = verify(test_image)
    if matched_uid:
        print(f"Matched Student: {matched_uid} - Proceed to Roll Number/Connection!")
    else:
        print("No match - Retry.")