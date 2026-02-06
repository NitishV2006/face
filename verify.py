import pickle
import cv2
import numpy as np
from utils import preprocess_image, get_embedding, compare_embeddings

# ====================== CONFIG ======================
THRESHOLD = 0.58          # 0.55 = more lenient, 0.65 = stricter
INFERENCE_EVERY_N_FRAMES = 4  # Higher = smoother, lower = more responsive
AUTO_STOP_ON_MATCH = True
# ===================================================

def load_db():
    try:
        with open('embeddings.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Error: embeddings.pkl not found. Run enroll.py first.")
        return {}
    except Exception as e:
        print(f"Error loading database: {e}")
        return {}

def verify_in_memory(frame_bgr):
    """Verify directly from webcam frame (numpy array)"""
    db = load_db()
    if not db:
        return None, 0.0, "No Database"

    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(frame_rgb)  # This may raise "No face detected"
        scan_emb = get_embedding(preprocessed)
        
        similarities = {}
        for uid, emb in db.items():
            sim = compare_embeddings(scan_emb, emb)
            similarities[uid] = sim

        if not similarities:
            return None, 0.0, "No Comparison"

        best_uid = max(similarities, key=similarities.get)
        best_score = similarities[best_uid]

        if best_score > THRESHOLD:
            return best_uid, best_score, "Matched"
        else:
            return None, best_score, "Unmatched"
    except ValueError as ve:
        if "No face detected" in str(ve):
            return None, 0.0, "No Face"
        else:
            print(f"Verification error: {ve}")
            return None, 0.0, "Error"
    except Exception as e:
        print(f"Verification error: {e}")
        return None, 0.0, "Error"

def live_scan():
    print("Starting Live Face Scan...")
    print("Press 'q' to quit\n")
    print("Will auto-close after 5 seconds if no match\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return

    start_time = cv2.getTickCount() / cv2.getTickFrequency()  # seconds
    TIMEOUT_SECONDS = 5.0

    frame_count = 0
    last_status = "Scanning..."
    last_color = (255, 255, 0)  # cyan
    last_score = 0.0
    last_uid = None

    while True:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - start_time

        # Auto-close after timeout if no match yet
        if elapsed >= TIMEOUT_SECONDS and last_uid is None:
            print(f"Timeout ({TIMEOUT_SECONDS}s) - No match detected. Closing.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_count += 1

        status = "Scanning..."
        color = (255, 255, 0)  # cyan

        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            uid, score, detection_status = verify_in_memory(frame)

            if detection_status == "No Face":
                status = "No Face Detected"
                color = (0, 0, 255)  # red

            elif detection_status == "Unmatched":
                status = "Face Unmatched"
                color = (0, 0, 255)  # red

            elif detection_status == "Matched":
                status = "Face Matched Successfully Verified"
                color = (0, 255, 0)  # green
                last_uid = uid
                last_score = score
                print(f"\nSUCCESS → Matched: {uid} (Score: {score:.3f})")

                # Show success message for 1.5–2 seconds before closing
                cv2.putText(frame, status, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                cv2.imshow("Academic Monitor - Live Face Scan", frame)
                cv2.waitKey(1800)  # 1.8 seconds
                break

            else:
                status = "Processing Error"
                color = (0, 165, 255)  # orange

            # Remember last status for in-between frames
            last_status = status
            last_color = color
            last_score = score if score > 0 else last_score

        # Display current/last status
        cv2.putText(frame, last_status, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, last_color, 4)
        cv2.putText(frame, f"Score: {last_score:.3f}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Optional: show countdown in corner
        remaining = max(0, TIMEOUT_SECONDS - elapsed)
        cv2.putText(frame, f"Auto-close in: {remaining:.1f}s", (frame.shape[1]-220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Academic Monitor - Live Face Scan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live Scan Stopped.")

if __name__ == "__main__":
    live_scan()