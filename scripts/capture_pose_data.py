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
    parser.add_argument(
        "--mirror",
        type=str,
        default="both",
        choices=["none", "display", "process", "both"],
        help="Lật gương trái–phải: none/display/process/both (mặc định both = selfie mode chuẩn)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Nếu file đã tồn tại thì gộp thêm dữ liệu thay vì ghi đè"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        print("❌ Không mở được camera. Kiểm tra --camera hoặc quyền truy cập.")
        sys.exit(1)

    print("Get ready!")
    time.sleep(5)
    print(f"Capturing pose data for '{args.pose_name}' (press 'q' to stop early)")

    start_time = time.time()
    pose_data = []

    # MediaPipe
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

    valid_count = 0
    total_count = 0
    no_face_count = 0
    last_console_tick = -1

    def _has_face(res):
        return (res is not None) and getattr(res, "multi_face_landmarks", None)

    while cap.isOpened():
        elapsed = time.time() - start_time
        if elapsed >= args.duration:
            print("End capturing (duration reached)")
            break

        success, frame_bgr = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        total_count += 1

        # Lật trước khi trích đặc trưng nếu cần
        if args.mirror in ("process", "both"):
            frame_bgr = cv2.flip(frame_bgr, 1)

        # RGB cho MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        face_results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)

        # Nếu không có mặt → bỏ frame (tránh crash trong extract_features)
        if not _has_face(face_results):
            no_face_count += 1
            feature = None
        else:
            # Bọc try/except để không crash nếu utils/feature_extraction gặp case lạ
            try:
                feature = extract_features(mp_hands, face_results, hand_results)
            except Exception:
                feature = None

        if feature is not None:
            arr = np.asarray(feature, dtype=float).ravel()
            if arr.size > 0 and np.all(np.isfinite(arr)):
                pose_data.append(arr)
                valid_count += 1

        # Vẽ landmarks
        frame_rgb.flags.writeable = True
        if _has_face(face_results):
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_rgb,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
        if hand_results and getattr(hand_results, "multi_hand_landmarks", None):
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_rgb,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Hiển thị
        view_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if args.mirror == "display":
            view_bgr = cv2.flip(view_bgr, 1)

        remaining = max(0.0, args.duration - elapsed)
        overlay_lines = [
            f"Class: {args.pose_name}",
            f"Frames (valid/total): {valid_count}/{total_count}",
            f"No-face frames: {no_face_count}",
            f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s",
            f"Mirror: {args.mirror} | Press 'q' to stop"
        ]
        y0, dy = 30, 22
        for i, line in enumerate(overlay_lines):
            y = y0 + i * dy
            cv2.putText(view_bgr, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view_bgr, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('MediaPipe Face and Hand Detection', view_bgr)

        # Log console mỗi ~1s
        tick = int(elapsed)
        if tick != last_console_tick:
            last_console_tick = tick
            print(f"[{tick:>3}s] valid/total = {valid_count}/{total_count}  "
                  f"({(valid_count/max(1,total_count))*100:.1f}% valid) | no_face={no_face_count}")

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print(f"Stopped by user at {elapsed:.1f}s — valid/total = {valid_count}/{total_count} | no_face={no_face_count}")
            break

    cap.release()
    cv2.destroyAllWindows()

    pose_data = np.array(pose_data, dtype=float)
    if pose_data.shape[0] == 0:
        print("⚠️ Không có khung hợp lệ nào. Không lưu file.")
        sys.exit(0)

    os.makedirs("data", exist_ok=True)
    out_path = f"data/{args.pose_name}.npy"

    # Gộp thêm nếu được yêu cầu
    if args.append and os.path.exists(out_path):
        try:
            old = np.load(out_path)
            pose_data = np.concatenate([old, pose_data], axis=0)
            print(f"🔁 Gộp thêm vào file cũ → tổng mới: {pose_data.shape[0]} mẫu")
        except Exception as e:
            print(f"⚠️ Không gộp được file cũ ({e}). Sẽ ghi đè bằng phiên thu hiện tại.")

    np.save(out_path, pose_data)
    print(f"✅ Save pose data successfully! shape={pose_data.shape} → {out_path}")
