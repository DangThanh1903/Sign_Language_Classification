import cv2
import mediapipe as mp
import time
import asyncio, edge_tts, tempfile, os
import base64
import streamlit as st
import streamlit.components.v1 as components
import warnings
import numpy as np
from collections import Counter, deque
from pathlib import Path

from utils.feature_extraction import *
from utils.strings import *
from utils.model import ASLClassificationModel
from config import MODEL_NAME, MODEL_CONFIDENCE

warnings.filterwarnings("ignore")

# ================== TTS (edge-tts) ==================
async def _edge_tts_generate(text: str, voice: str = "vi-VN-HoaiMyNeural"):
    if not text:
        return b""
    tmp_path = os.path.join(tempfile.gettempdir(), "tts_tmp_streamlit.mp3")
    tts = edge_tts.Communicate(text, voice=voice)
    await tts.save(tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return data

def tts_bytes_edge(text: str):
    try:
        data = asyncio.run(_edge_tts_generate(text))
        return data, "audio/mp3"
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(_edge_tts_generate(text))
        loop.close()
        return data, "audio/mp3"
    except Exception as e:
        print("edge-tts error:", e)
        return b"", "audio/mp3"

def play_audio_autoplay(audio_bytes: bytes, mime: str = "audio/mp3", sig: str = ""):
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <audio autoplay>
      <source src="data:{mime};base64,{b64}#{sig}" type="{mime}">
    </audio>
    """
    components.html(html, height=0)

# ================== MediaPipe ==================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ===== Tham số cơ bản =====
STABLE_SECONDS = 0.5
DEFAULT_SKIP_LABELS = {
    "binh_thuong", "bình_thường", "binh thuong", "bình thường", "ngồi yên", "ngoi yen", ""
}

# === [SENTENCE TIMING MODE] ===
WORD_MIN_SEC = 0.5
WORD_MAX_SEC = 2.0
SILENCE_FINALIZE_SEC = 3.0

# === CHỐNG LẶP TỪ ===
SAME_WORD_COOLDOWN_SEC = 1.2
MAX_CONSECUTIVE_DUP_COMPRESS = 3
REENTRY_GAP_DEFAULT_SEC = 0.3

# ================== An toàn cho feature extraction ==================
def safe_extract_features(mp_hands, face_results, hand_results):
    try:
        f = extract_features(mp_hands, face_results, hand_results)
        if f is None:
            return None
        arr = np.asarray(f, dtype=float).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr
    except Exception:
        return None

# ================== Từ điển & ghép câu ==================
LEXICON = {
    "xin_chao": "xin chào",
    "hello": "xin chào",
    "toi": "tôi",
    "ban": "bạn",
    "cam_on": "cảm ơn",
    "xin_loi": "xin lỗi",
    "toi_la": "tôi là",
}
PUNCT_TOKENS = {".", ",", "?", "!"}

def normalize_token(label: str) -> str:
    if not label:
        return ""
    if label in LEXICON:
        return LEXICON[label]
    return label.replace("_", " ")

def detok_vietnamese(tokens):
    out = []
    for t in tokens:
        if not t:
            continue
        if t in PUNCT_TOKENS:
            if out:
                out[-1] = out[-1].rstrip()
            out.append(t + " ")
        else:
            if out and not out[-1].endswith(" "):
                out.append(" ")
            out.append(t)
    s = "".join(out).strip()
    if s:
        s = s[0].upper() + s[1:]
    s = s.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    return s

# ================== Majority vote smoother ==================
class MajoritySmoother:
    def __init__(self, window_size=7):
        self.win = deque(maxlen=max(3, int(window_size)))

    def feed(self, label: str) -> str:
        self.win.append(label)
        if len(self.win) == 0:
            return label
        counts = Counter(self.win)
        most_common = counts.most_common()
        if len(most_common) == 0:
            return label
        top_count = most_common[0][1]
        candidates = [lab for lab, c in most_common if c == top_count]
        for lab in reversed(self.win):
            if lab in candidates:
                return lab
        return label

# ================== Gom từ theo thời gian + chống lặp ==================
class PhraseAssemblerTiming:
    def __init__(self, skip_labels=None, reentry_gap_sec=REENTRY_GAP_DEFAULT_SEC):
        self.tokens = []
        self.skip = set(skip_labels or [])
        self._hold_committed = False
        self.last_commit_time = 0.0
        self.last_commit_label = None
        self.reentry_gap_sec = float(reentry_gap_sec)
        self.last_leave_ts = {}

    def reset_hold_flag(self):
        self._hold_committed = False

    def mark_leave(self, label: str, now_ts: float):
        if label:
            self.last_leave_ts[label] = now_ts

    def _can_commit(self, label: str, now_ts: float) -> bool:
        if not label or (label in self.skip):
            return False
        if (self.last_commit_label == label) and ((now_ts - self.last_commit_time) < SAME_WORD_COOLDOWN_SEC):
            return False
        last_leave = self.last_leave_ts.get(label, None)
        if last_leave is not None and (now_ts - last_leave) < self.reentry_gap_sec:
            return False
        return True

    def _commit(self, label: str, now_ts: float):
        tok = normalize_token(label)
        if self.tokens and self.tokens[-1] == tok:
            return False
        self.tokens.append(tok)
        self.last_commit_label = label
        self.last_commit_time = now_ts
        return True

    def commit_word_if_valid(self, label: str, hold_sec: float, now_ts: float):
        if WORD_MIN_SEC <= hold_sec <= WORD_MAX_SEC and self._can_commit(label, now_ts):
            return self._commit(label, now_ts)
        return False

    def commit_on_overflow(self, label: str, hold_sec: float, now_ts: float):
        if hold_sec > WORD_MAX_SEC and (not self._hold_committed) and self._can_commit(label, now_ts):
            ok = self._commit(label, now_ts)
            if ok:
                self._hold_committed = True
            return ok
        return False

    def _compress_tail_duplicates(self):
        if not self.tokens:
            return
        while len(self.tokens) >= 2 and self.tokens[-1] == self.tokens[-2]:
            self.tokens.pop()

    def finalize_if_silence(self, silence_sec: float):
        if silence_sec >= SILENCE_FINALIZE_SEC and self.tokens:
            self._compress_tail_duplicates()
            if not self.tokens or self.tokens[-1] in {".", ",", "?", "!"}:
                pass
            else:
                self.tokens.append(".")
            sentence = detok_vietnamese(self.tokens)
            self.tokens.clear()
            return sentence
        return None

    def partial_text(self):
        return detok_vietnamese(self.tokens)

# ================== [SPELLING MODE] ==================
LABEL_TO_LETTER = {
    **{chr(o): chr(o) for o in range(ord('a'), ord('z')+1)},
    **{chr(o): chr(o) for o in range(ord('A'), ord('Z')+1)},
    "đ": "đ", "Đ": "Đ",
}
LABEL_TO_TONE = {
    "dau_sac": "sac",
    "dau_huyen": "huyen",
    "dau_hoi": "hoi",
    "dau_nga": "nga",
    "dau_nang": "nang",
    "dau_ngang": "ngang",
}
TONE_MAP = {
    "a": {"sac":"á","huyen":"à","hoi":"ả","nga":"ã","nang":"ạ","ngang":"a"},
    "ă":{"sac":"ắ","huyen":"ằ","hoi":"ẳ","nga":"ẵ","nang":"ặ","ngang":"ă"},
    "â":{"sac":"ấ","huyen":"ầ","hoi":"ẩ","nga":"ẫ","nang":"ậ","ngang":"â"},
    "e": {"sac":"é","huyen":"è","hoi":"ẻ","nga":"ẽ","nang":"ẹ","ngang":"e"},
    "ê":{"sac":"ế","huyen":"ề","hoi":"ể","nga":"ễ","nang":"ệ","ngang":"ê"},
    "i": {"sac":"í","huyen":"ì","hoi":"ỉ","nga":"ĩ","nang":"ị","ngang":"i"},
    "o": {"sac":"ó","huyen":"ò","hoi":"ỏ","nga":"õ","nang":"ọ","ngang":"o"},
    "ô":{"sac":"ố","huyen":"ồ","hoi":"ổ","nga":"ỗ","nang":"ộ","ngang":"ô"},
    "ơ":{"sac":"ớ","huyen":"ờ","hoi":"ở","nga":"ỡ","nang":"ợ","ngang":"ơ"},
    "u": {"sac":"ú","huyen":"ù","hoi":"ủ","nga":"ũ","nang":"ụ","ngang":"u"},
    "ư":{"sac":"ứ","huyen":"ừ","hoi":"ử","nga":"ữ","nang":"ự","ngang":"ư"},
    "y": {"sac":"ý","huyen":"ỳ","hoi":"ỷ","nga":"ỹ","nang":"ỵ","ngang":"y"},
}
VOWEL_PRIORITY = ["a","ă","â","e","ê","o","ô","ơ","ư","i","y"]

def apply_tone_to_word(base_word: str, tone: str) -> str:
    if not base_word or tone == "ngang" or tone not in {"sac","huyen","hoi","nga","nang"}:
        return base_word
    lower = base_word.lower()
    candidate_idx = None
    for v in VOWEL_PRIORITY:
        idx = lower.rfind(v)
        if idx != -1:
            candidate_idx = idx
            break
    if candidate_idx is None:
        return base_word
    orig_char = base_word[candidate_idx]
    lv = lower[candidate_idx]
    if lv not in TONE_MAP:
        return base_word
    rep = TONE_MAP[lv][tone]
    if orig_char.isupper():
        rep = rep.upper()
    return base_word[:candidate_idx] + rep + base_word[candidate_idx+1:]

class Speller:
    def __init__(self):
        self.letters = []
        self.pending_tone = "ngang"

    def feed_label(self, label: str):
        if not label:
            return
        lab = label.strip()
        if lab in {"xoa", "backspace"}:
            if self.letters:
                self.letters.pop()
            return
        tone = LABEL_TO_TONE.get(lab)
        if tone:
            self.pending_tone = tone
            return
        ch = LABEL_TO_LETTER.get(lab)
        if ch:
            self.letters.append(ch.lower())

    def partial_word(self) -> str:
        base = "".join(self.letters)
        return apply_tone_to_word(base, self.pending_tone)

    def finalize(self) -> str | None:
        base = "".join(self.letters).strip()
        if not base:
            self.pending_tone = "ngang"
            return None
        word = apply_tone_to_word(base, self.pending_tone)
        self.letters.clear()
        self.pending_tone = "ngang"
        return word

# ================== MAIN ==================
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
            .big-font {
                color: #e76f51 !important;
                font-size: 60px !important;
                border: 0.5rem solid #fcbf49 !important;
                border-radius: 2rem;
                text-align: center;
            }
            .small { font-size: 14px; color: #888; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar: tuỳ chọn
    st.sidebar.header("Tuỳ chọn")
    ENABLE_AUTO_SENTENCE = st.sidebar.toggle("🧠 Tự cảm nhận & dịch cả câu (0.5–2s, im 3s chốt)", value=True)
    ENABLE_SPELLING_MODE = st.sidebar.toggle("🅰️ Spelling mode (đánh vần từng âm tiết)", value=False)
    TTS_WORD_MODE = st.sidebar.toggle("🔊 Đọc từng từ trong khi ghép câu", value=False)

    ENABLE_SMOOTHING = st.sidebar.toggle("🪄 Bật bộ lọc số đông (majority vote)", value=True)
    SMOOTH_WINDOW = st.sidebar.slider("Cửa sổ majority (frame)", min_value=3, max_value=15, value=7, step=2)
    REENTRY_GAP_SEC = st.sidebar.slider("Khoảng rời từ cũ (re-entry gap, giây)", min_value=0.0, max_value=1.0, value=REENTRY_GAP_DEFAULT_SEC, step=0.05)

    # === Unknown theo ngưỡng → map về nhãn skip ===
    UNKNOWN_THRESHOLD = st.sidebar.slider(
        "🔒 Ngưỡng Unknown (max proba < ngưỡng → coi như 'ngồi yên')",
        min_value=0.0, max_value=1.0, value=0.60, step=0.01
    )
    UNKNOWN_LABEL = st.sidebar.selectbox(
        "🏷️ Nhãn dùng cho Unknown",
        options=["binh_thuong", "bình_thường", "ngồi yên"],
        index=0
    )

    # === Chọn model .pkl + confidence + camera ===
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    available_models = sorted([p.name for p in models_dir.glob("*.pkl")])
    default_model_name = MODEL_NAME if MODEL_NAME in available_models else (available_models[0] if available_models else "<không có model>")

    selected_model = st.sidebar.selectbox(
        "📦 Chọn mô hình (.pkl) trong /models",
        options=available_models if available_models else ["<không có model>"],
        index=available_models.index(default_model_name) if available_models and default_model_name in available_models else 0
    )

    side_conf = st.sidebar.slider(
        "🎯 Độ tự tin MediaPipe (min_detection/tracking_confidence)",
        min_value=0.1, max_value=0.9, value=float(MODEL_CONFIDENCE), step=0.05
    )

    cam_index = st.sidebar.number_input("📷 Camera index", min_value=0, value=0, step=1)
    show_fps = st.sidebar.toggle("⏱️ Hiển thị FPS", value=True)
    proba_chart = st.sidebar.toggle("📊 Hiển thị xác suất lớp (nếu model hỗ trợ)", value=False)

    # Quick actions
    if st.sidebar.button("🧹 Reset câu tạm"):
        st.session_state["reset_phrase_tokens"] = True
    else:
        st.session_state["reset_phrase_tokens"] = False

    if st.sidebar.button("🔊 Đọc câu tạm"):
        st.session_state["speak_partial"] = True
    else:
        st.session_state["speak_partial"] = False

    col1, col2 = st.columns([4, 2])
    with col1:
        video_placeholder = st.empty()
    with col2:
        prediction_placeholder = st.empty()
        sentence_placeholder = st.empty()

    if "audio_sig" not in st.session_state:
        st.session_state.audio_sig = 0
    if "silence_since" not in st.session_state:
        st.session_state.silence_since = None

    cap = cv2.VideoCapture(int(cam_index))

    expression_handler = ExpressionHandler()

    # Assembler & smoother
    phrase_timing = PhraseAssemblerTiming(skip_labels=DEFAULT_SKIP_LABELS, reentry_gap_sec=REENTRY_GAP_SEC)
    smoother = MajoritySmoother(window_size=SMOOTH_WINDOW) if ENABLE_SMOOTHING else None
    speller = Speller()

    print("Initialising model ...")
    if selected_model == "<không có model>":
        st.error("Chưa có mô hình trong thư mục /models. Hãy đặt file .pkl vào đó (ví dụ từ train.py).")
        st.stop()
    try:
        model = ASLClassificationModel.load_model(models_dir / selected_model)
    except Exception as e:
        st.exception(e)
        st.stop()

    # MediaPipe modules
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=side_conf,
        min_tracking_confidence=side_conf
    )
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=side_conf,
        min_tracking_confidence=side_conf
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    print("Starting application")

    # Trạng thái nhãn ổn định
    current_label = None
    current_since = 0.0

    # FPS
    prev_ts = time.time()
    fps_placeholder = st.sidebar.empty()

    while cap.isOpened():
        ok, image = cap.read()
        if not ok:
            print("Ignoring empty camera frame.")
            continue

        # FPS
        if show_fps:
            now_ts_fps = time.time()
            dt = now_ts_fps - prev_ts
            prev_ts = now_ts_fps
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_placeholder.markdown(f"**FPS:** {fps:.1f}")

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(image)
        hand_results = hands.process(image)

        feature = safe_extract_features(mp_hands, face_results, hand_results)

        # ===== Dự đoán với Unknown theo ngưỡng =====
        expression = None
        if feature is not None:
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([feature])[0]
                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(probs[pred_idx])
                    classes = getattr(model, "classes_", None)
                    pred_label = classes[pred_idx] if classes is not None else str(pred_idx)
                    expression = UNKNOWN_LABEL if pred_conf < UNKNOWN_THRESHOLD else pred_label
                else:
                    expression = model.predict(feature)
            except Exception:
                expression = None
        else:
            expression = None

        if expression is not None:
            expression_handler.receive(expression)
            raw_text = expression_handler.get_message() or ""
        else:
            expression_handler.receive("")
            raw_text = ""

        display_label = (raw_text or "").strip()
        norm_label = display_label.lower()

        # Majority vote smoothing (nếu bật)
        if smoother is not None:
            norm_label = smoother.feed(norm_label)

        now = time.time()

        # Theo dõi im lặng
        if norm_label in DEFAULT_SKIP_LABELS:
            if st.session_state.silence_since is None:
                st.session_state.silence_since = now
        else:
            st.session_state.silence_since = None

        # Cho phép reset câu tạm
        if st.session_state.get("reset_phrase_tokens", False):
            phrase_timing.tokens.clear()

        # Phát lại câu tạm
        if st.session_state.get("speak_partial", False):
            text = phrase_timing.partial_text()
            if text:
                audio_bytes, mime = tts_bytes_edge(text)
                if audio_bytes:
                    st.session_state.audio_sig += 1
                    play_audio_autoplay(audio_bytes, mime, sig=str(st.session_state.audio_sig))

        # ======= Dùng prev_label để commit & TTS từ =======
        prev_label = current_label
        if norm_label != (current_label or ""):
            hold_sec = (now - current_since) if current_since else 0.0

            if ENABLE_AUTO_SENTENCE and prev_label and (prev_label not in DEFAULT_SKIP_LABELS):
                if ENABLE_SPELLING_MODE:
                    speller.feed_label(prev_label)
                    committed = False
                else:
                    committed = phrase_timing.commit_word_if_valid(prev_label, hold_sec, now)

                phrase_timing.mark_leave(prev_label, now)
                phrase_timing.reset_hold_flag()
            else:
                committed = False
                if prev_label:
                    phrase_timing.mark_leave(prev_label, now)

            current_label = norm_label
            current_since = now

            # Đọc từ vừa chốt (nếu bật)
            if TTS_WORD_MODE and committed:
                word_text = normalize_token(prev_label)
                audio_bytes, mime = tts_bytes_edge(word_text)
                if audio_bytes:
                    st.session_state.audio_sig += 1
                    play_audio_autoplay(audio_bytes, mime, sig=str(st.session_state.audio_sig))

        # Nếu nhãn không đổi, kiểm tra overflow > WORD_MAX_SEC để chốt 1 lần
        elapsed = time.time() - current_since if current_since else 0.0
        if ENABLE_AUTO_SENTENCE:
            if not ENABLE_SPELLING_MODE:
                phrase_timing.commit_on_overflow(current_label, elapsed, now)

            # Nếu im lặng đủ lâu -> finalize
            if st.session_state.silence_since is not None:
                silence_sec = now - st.session_state.silence_since
                if ENABLE_SPELLING_MODE:
                    word = speller.finalize()
                    if word:
                        phrase_timing.tokens.append(word)

                final_sentence = phrase_timing.finalize_if_silence(silence_sec)
                if final_sentence:
                    audio_bytes, mime = tts_bytes_edge(final_sentence)
                    if audio_bytes:
                        st.session_state.audio_sig += 1
                        play_audio_autoplay(audio_bytes, mime, sig=str(st.session_state.audio_sig))
                    prediction_placeholder.markdown(
                        f'''<h2 class="big-font">{final_sentence}</h2><p class="small">[auto-finalized after {SILENCE_FINALIZE_SEC:.1f}s silence]</p>''',
                        unsafe_allow_html=True
                    )

            # Hiển thị câu/âm tiết tạm thời
            if ENABLE_SPELLING_MODE:
                partial_spell = speller.partial_word()
                sentence_placeholder.markdown(
                    f'''<p class="small">Âm tiết: <b>{partial_spell}</b><br/>Câu: <b>{phrase_timing.partial_text()}</b></p>''',
                    unsafe_allow_html=True
                )
            else:
                sentence_placeholder.markdown(
                    f'''<p class="small">Câu: <b>{phrase_timing.partial_text()}</b></p>''',
                    unsafe_allow_html=True
                )

        # (Tuỳ chọn) hiển thị xác suất lớp
        if proba_chart and feature is not None and hasattr(model, "predict_proba"):
            try:
                classes = getattr(model, "classes_", None)
                probs = model.predict_proba([feature])[0]
                if classes is None:
                    classes = [f"class_{i}" for i in range(len(probs))]
                order = np.argsort(probs)[::-1][:5]
                top = [(classes[i], float(probs[i])) for i in order]
                st.sidebar.write({k: v for k, v in top})
            except Exception:
                pass

        # Vẽ landmarks
        if face_results and getattr(face_results, "multi_face_landmarks", None):
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
        if hand_results and getattr(hand_results, "multi_hand_landmarks", None):
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Hiển thị video + nhãn hiện tại
        video_placeholder.image(image, channels="RGB", use_column_width=True)
        debug = f"(hold {elapsed:.2f}s) smoothing:{'on' if smoother else 'off'} re-gap:{REENTRY_GAP_SEC:.2f}s spelling:{'on' if ENABLE_SPELLING_MODE else 'off'}"
        prediction_placeholder.markdown(
            f'''<h2 class="big-font">{current_label or ''}</h2><p class="small">{debug}</p>''',
            unsafe_allow_html=True
        )

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
