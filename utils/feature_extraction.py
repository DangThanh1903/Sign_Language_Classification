import numpy as np
from config import *

def extract_single_hand(mp_hands, hand_landmarks):
    # 21 landmark, lấy (x, y) chuẩn hoá theo ảnh (MediaPipe đã chuẩn hoá 0..1)
    arr = np.zeros((21, 2), dtype=float)

    def get_xy(lm):
        if lm is None:
            return np.array([0.0, 0.0])
        return np.array([lm.x, lm.y], dtype=float)

    arr[0]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    arr[1]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC])
    arr[2]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP])
    arr[3]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP])
    arr[4]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
    arr[5]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP])
    arr[6]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    arr[7]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP])
    arr[8]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    arr[9]  = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
    arr[10] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
    arr[11] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP])
    arr[12] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    arr[13] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    arr[14] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP])
    arr[15] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP])
    arr[16] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP])
    arr[17] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])
    arr[18] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP])
    arr[19] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP])
    arr[20] = get_xy(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])
    return arr

def extract_hand_result(mp_hands, hand_results):
    """
    Trả về vector tay: [Left(21x2), Right(21x2)] → flatten (84)
    Nếu không có tay → zeros(84)
    Nếu 1 tay → đúng vị trí (left/right) + zeros cho tay còn lại.
    """
    if hand_results is None or getattr(hand_results, "multi_hand_landmarks", None) is None:
        return np.zeros(FEATURES_PER_HAND * 4, dtype=float)  # 84

    hands_lm = hand_results.multi_hand_landmarks
    handed = getattr(hand_results, "multi_handedness", None)

    # Không tin thứ tự hands_lm; map theo handedness để tránh nhầm
    left_arr = np.zeros((FEATURES_PER_HAND, 2), dtype=float)
    right_arr = np.zeros((FEATURES_PER_HAND, 2), dtype=float)

    if handed is None:
        # fallback: nếu không có thông tin trái/phải → đặt vào left, right=0
        h0 = extract_single_hand(mp_hands, hands_lm[0])
        left_arr = h0
    else:
        for lm, hd in zip(hands_lm, handed):
            label = hd.classification[0].label  # "Left" hoặc "Right"
            if label == "Left":
                left_arr = extract_single_hand(mp_hands, lm)
            elif label == "Right":
                right_arr = extract_single_hand(mp_hands, lm)

    # ghép Left trước, Right sau để nhất quán
    both = np.hstack((left_arr, right_arr)).flatten()  # (21,4) -> 84
    return both.astype(float)

def extract_face_result(face_results):
    """
    Trả về (x_mean, y_mean) của tất cả landmarks mặt.
    Nếu không có mặt → zeros(2) để giữ kích thước ổn định.
    """
    if face_results is None or getattr(face_results, "multi_face_landmarks", None) is None:
        return np.zeros(2, dtype=float)

    faces = face_results.multi_face_landmarks
    if len(faces) == 0 or faces[0] is None:
        return np.zeros(2, dtype=float)

    face = faces[0]
    face_xy = np.array([[lm.x, lm.y] for lm in face.landmark], dtype=float)
    if face_xy.size == 0 or not np.all(np.isfinite(face_xy)):
        return np.zeros(2, dtype=float)

    return np.mean(face_xy, axis=0)  # (2,)

def extract_features(mp_hands, face_results, hand_results):
    """
    Hợp nhất: face(2) + hands(84) = 86 chiều
    """
    face_features = extract_face_result(face_results)           # (2,)
    hand_features = extract_hand_result(mp_hands, hand_results) # (84,)
    feat = np.hstack((face_features, hand_features)).astype(float)
    return feat
