import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import mediapipe as mp
import argparse
import time
import numpy as np
from utils.feature_extraction import *

import warnings
warnings.filterwarnings("ignore")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Data Capture")
    parser.add_argument("--pose_name", type=str, default="test", help="Tên lớp/pose (file .npy)")
    parser.add_argument("--confidence", type=float, default=0.6, help="Ngưỡng MediaPipe")
    parser.add_argument("--duration", type=int, default=60, help="Thời lượng ghi (giây)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.camera))

    print("Get ready!")
    time.sleep(5)
    print(f"Capturing pose data for '{args.pose_name}' (press 'q' to stop early)")

    start_time = time.time()
    pose_data = []

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    )
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    valid_count = 0          # số frame hợp lệ (có feature tốt)
    total_count = 0          # tổng frame đọc được
    last_console_tick = -1   # để log ra console mỗi 1s

    while cap.isOpened():
        elapsed = time.time() - start_time
        if elapsed >= args.duration:
            print("End capturing (duration reached)")
            break

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        total_count += 1

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(image)
        hand_results = hands.process(image)

        feature = extract_features(mp_hands, face_results, hand_results)
        if feature is not None:
            arr = np.asarray(feature, dtype=float).ravel()
            if arr.size > 0 and np.all(np.isfinite(arr)):
                pose_data.append(arr)
                valid_count += 1

        # Vẽ landmarks
        image.flags.writeable = True
        if face_results and getattr(face_results, "multi_face_landmarks", None):
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
        if hand_results and getattr(hand_results, "multi_hand_landmarks", None):
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # ===== Overlay tiến độ lên preview =====
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        remaining = max(0.0, args.duration - elapsed)
        overlay_lines = [
            f"Class: {args.pose_name}",
            f"Frames (valid/total): {valid_count}/{total_count}",
            f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s",
            "Press 'q' to stop"
        ]
        y0, dy = 30, 22
        for i, line in enumerate(overlay_lines):
            y = y0 + i * dy
            cv2.putText(image_bgr, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)  # viền đen
            cv2.putText(image_bgr, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)  # chữ vàng

        cv2.imshow('MediaPipe Face and Hand Detection', cv2.flip(image_bgr, 1))

        # ===== Log ra console mỗi ~1 giây =====
        tick = int(elapsed)
        if tick != last_console_tick:
            last_console_tick = tick
            print(f"[{tick:>3}s] valid/total = {valid_count}/{total_count}  "
                  f"({(valid_count/max(1,total_count))*100:.1f}% valid)")

        # Nhấn q để dừng sớm
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print(f"Stopped by user at {elapsed:.1f}s — valid/total = {valid_count}/{total_count}")
            break

    cap.release()
    cv2.destroyAllWindows()

    pose_data = np.array(pose_data, dtype=float)
    if pose_data.shape[0] == 0:
        print("⚠️ Không có khung hợp lệ nào. Không lưu file.")
    else:
        os.makedirs("data", exist_ok=True)
        out_path = f"data/{args.pose_name}.npy"
        np.save(out_path, pose_data)
        print(f"✅ Save pose data successfully! shape={pose_data.shape} → {out_path}")
