"""
Real-Time Sleep Detection System
Production-grade, mathematically rigorous, single-file implementation.

Requirements:
    pip install opencv-python mediapipe numpy

Model download:
    wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
         -O face_landmarker.task
"""

# ============================================================
# IMPORTS
# ============================================================
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import math
import os
import sys
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "face_landmarker.task"
CAMERA_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Calibration
CALIBRATION_DURATION = 3.0  # seconds

# Sliding window
WINDOW_SECONDS = 2.0
WINDOW_FRAMES = int(WINDOW_SECONDS * TARGET_FPS)  # ~60

# EAR / openness thresholds
CLOSED_OPENNESS_THRESHOLD = 0.25  # O(t) < 0.25 => closed

# Blink classification (seconds)
BLINK_NORMAL_MAX = 0.40
BLINK_LONG_MAX = 1.50

# Motion entropy
MOTION_LOW_THRESHOLD = 0.30  # M(t) relative to baseline
ENTROPY_LOW_THRESHOLD = 0.60

# EAR slope (fatigue)
EAR_SLOPE_FATIGUE = -0.004   # dE/dt per frame threshold

# Head velocity / nod
NOD_VELOCITY_THRESHOLD = 0.008   # normalized units
NOD_LINGER_FRAMES = 10

# Confidence weights (must sum to 1.0)
W_PERCLOS = 0.45
W_LONG_CLOSURE = 0.25
W_LOW_MOTION = 0.10
W_NOD = 0.10
W_EAR_DROP = 0.10

# EMA
ALPHA_EMA = 0.40

# State machine thresholds
TH_ALERT_TO_TIRED = 0.45
TH_TIRED_TO_DROWSY = 0.60
TH_DROWSY_TO_MICROSLEEP = 0.75

TH_MICROSLEEP_EXIT = 0.50
TH_DROWSY_EXIT = 0.35

HOLD_ALERT_TO_TIRED = 1.0       # seconds
HOLD_TIRED_TO_DROWSY = 1.0
HOLD_DROWSY_TO_MICROSLEEP = 1.5
HOLD_MICROSLEEP_EXIT = 1.0
HOLD_DROWSY_EXIT = 2.0

# MediaPipe landmark indices
# Left eye: upper/lower vertical pair
LEFT_EYE_UPPER = [159, 158, 157]
LEFT_EYE_LOWER = [145, 153, 154]
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

# Right eye: upper/lower vertical pair
RIGHT_EYE_UPPER = [386, 385, 384]
RIGHT_EYE_LOWER = [374, 380, 381]
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

NOSE_TIP = 1

# ============================================================
# STATE DEFINITIONS
# ============================================================
STATE_ALERT = "ALERT"
STATE_TIRED = "TIRED"
STATE_DROWSY = "DROWSY"
STATE_MICROSLEEP = "MICRO_SLEEP"

STATE_COLORS = {
    STATE_ALERT: (0, 220, 0),
    STATE_TIRED: (0, 200, 255),
    STATE_DROWSY: (0, 100, 255),
    STATE_MICROSLEEP: (0, 0, 220),
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def landmark_to_np(lm, w, h):
    """Convert normalized landmark to pixel coords (x, y)."""
    return np.array([lm.x * w, lm.y * h])


def eye_aspect_ratio(upper_pts, lower_pts, left_pt, right_pt):
    """
    Compute EAR:
        EAR = (mean vertical distance) / (horizontal distance)
    """
    v_dists = [np.linalg.norm(u - l) for u, l in zip(upper_pts, lower_pts)]
    v_mean = np.mean(v_dists)
    h_dist = np.linalg.norm(right_pt - left_pt) + 1e-6
    return v_mean / h_dist


def vertical_eye_opening(upper_pts, lower_pts):
    """Mean vertical opening distance in pixels."""
    return np.mean([np.linalg.norm(u - l) for u, l in zip(upper_pts, lower_pts)])


def extract_eye_landmarks(landmarks, w, h):
    """Return upper, lower, left, right np arrays for both eyes."""
    def lm(idx):
        return landmark_to_np(landmarks[idx], w, h)

    left_upper = [lm(i) for i in LEFT_EYE_UPPER]
    left_lower = [lm(i) for i in LEFT_EYE_LOWER]
    left_left = lm(LEFT_EYE_LEFT)
    left_right = lm(LEFT_EYE_RIGHT)

    right_upper = [lm(i) for i in RIGHT_EYE_UPPER]
    right_lower = [lm(i) for i in RIGHT_EYE_LOWER]
    right_left = lm(RIGHT_EYE_LEFT)
    right_right = lm(RIGHT_EYE_RIGHT)

    return (left_upper, left_lower, left_left, left_right,
            right_upper, right_lower, right_left, right_right)


def compute_landmark_motion(prev_lms, curr_lms, w, h):
    """
    Mean per-landmark Euclidean displacement (normalized by frame diagonal).
    """
    if prev_lms is None or curr_lms is None:
        return 0.0
    diag = math.sqrt(w * w + h * h) + 1e-6
    displacements = []
    n = min(len(prev_lms), len(curr_lms))
    for i in range(n):
        p = landmark_to_np(prev_lms[i], w, h)
        c = landmark_to_np(curr_lms[i], w, h)
        displacements.append(np.linalg.norm(c - p) / diag)
    return float(np.mean(displacements)) if displacements else 0.0


def safe_mean(arr, fallback=1.0):
    if len(arr) == 0:
        return fallback
    m = np.mean(arr)
    return float(m) if not np.isnan(m) else fallback


def safe_std(arr, fallback=0.0):
    if len(arr) < 2:
        return fallback
    s = np.std(arr)
    return float(s) if not np.isnan(s) else fallback


# ============================================================
# ALARM
# ============================================================

class AlarmController:
    """Non-blocking beep alarm using a dedicated thread."""

    def __init__(self):
        self._active = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        with self._lock:
            if not self._active:
                self._active = True
                self._thread = threading.Thread(
                    target=self._beep_loop, daemon=True)
                self._thread.start()

    def stop(self):
        with self._lock:
            self._active = False

    def _beep_loop(self):
        while True:
            with self._lock:
                if not self._active:
                    break
            # Platform-safe beep
            try:
                if sys.platform.startswith("linux"):
                    os.system("play -n synth 0.3 sin 880 vol 0.5 2>/dev/null &")
                elif sys.platform == "darwin":
                    os.system("afplay /System/Library/Sounds/Ping.aiff &")
                elif sys.platform == "win32":
                    import winsound
                    winsound.Beep(880, 300)
            except Exception:
                pass
            time.sleep(0.5)

    def is_active(self):
        with self._lock:
            return self._active


# ============================================================
# CALIBRATION
# ============================================================

class Calibration:
    __slots__ = [
        "baseline_ear", "baseline_eye_opening",
        "baseline_motion", "complete",
        "_ear_samples", "_opening_samples",
        "_motion_samples", "_start_time"
    ]

    def __init__(self):
        self.baseline_ear = 0.30
        self.baseline_eye_opening = 10.0
        self.baseline_motion = 0.002
        self.complete = False
        self._ear_samples = []
        self._opening_samples = []
        self._motion_samples = []
        self._start_time = None

    def reset(self):
        self.__init__()

    def update(self, ear, opening, motion, now):
        if self._start_time is None:
            self._start_time = now
        elapsed = now - self._start_time
        if elapsed < CALIBRATION_DURATION:
            self._ear_samples.append(ear)
            self._opening_samples.append(opening)
            if motion > 0:
                self._motion_samples.append(motion)
            return False  # not done
        else:
            if not self.complete:
                self.baseline_ear = safe_mean(self._ear_samples, 0.30)
                self.baseline_eye_opening = safe_mean(self._opening_samples, 10.0)
                self.baseline_motion = safe_mean(self._motion_samples, 0.002)
                self.baseline_ear = max(self.baseline_ear, 1e-4)
                self.baseline_eye_opening = max(self.baseline_eye_opening, 1.0)
                self.baseline_motion = max(self.baseline_motion, 1e-5)
                self.complete = True
            return True

    def progress(self, now):
        if self._start_time is None:
            return 0.0
        return min(1.0, (now - self._start_time) / CALIBRATION_DURATION)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureExtractor:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.prev_landmarks = None
        self.prev_nose_y = None

    def extract(self, landmarks, calib: Calibration):
        w, h = self.w, self.h

        (l_up, l_lo, l_lt, l_rt,
         r_up, r_lo, r_lt, r_rt) = extract_eye_landmarks(landmarks, w, h)

        # EAR
        ear_l = eye_aspect_ratio(l_up, l_lo, l_lt, l_rt)
        ear_r = eye_aspect_ratio(r_up, r_lo, r_lt, r_rt)
        ear = (ear_l + ear_r) / 2.0

        # Vertical opening
        op_l = vertical_eye_opening(l_up, l_lo)
        op_r = vertical_eye_opening(r_up, r_lo)
        opening = (op_l + op_r) / 2.0

        # Motion
        motion = compute_landmark_motion(
            self.prev_landmarks, landmarks, w, h)
        self.prev_landmarks = landmarks

        # Nose vertical velocity
        nose_y = landmarks[NOSE_TIP].y  # normalized [0,1]
        nose_vel = 0.0
        if self.prev_nose_y is not None:
            nose_vel = nose_y - self.prev_nose_y
        self.prev_nose_y = nose_y

        # Normalized features
        E_norm = ear / calib.baseline_ear
        O_norm = opening / calib.baseline_eye_opening
        M_norm = motion / calib.baseline_motion

        return {
            "ear": ear,
            "opening": opening,
            "motion": motion,
            "nose_vel": nose_vel,
            "E_norm": E_norm,
            "O_norm": O_norm,
            "M_norm": M_norm,
        }


# ============================================================
# TEMPORAL MODELING
# ============================================================

class TemporalModel:
    def __init__(self):
        self.ear_buf = deque(maxlen=WINDOW_FRAMES)
        self.openness_buf = deque(maxlen=WINDOW_FRAMES)
        self.motion_buf = deque(maxlen=WINDOW_FRAMES)
        self.ts_buf = deque(maxlen=WINDOW_FRAMES)
        self.nose_vel_buf = deque(maxlen=WINDOW_FRAMES)

        # Blink tracking
        self._in_blink = False
        self._blink_start = None
        self.blink_durations = deque(maxlen=30)
        self.long_blink_durations = deque(maxlen=30)

        # Nod tracking
        self._nod_counter = 0
        self._nod_flag = False

        # EAR slope buffer for finite difference
        self._ear_slope_buf = deque(maxlen=10)

    def update(self, feats: dict, now: float):
        O = feats["O_norm"]
        E = feats["E_norm"]
        M = feats["M_norm"]
        v = feats["nose_vel"]

        self.ear_buf.append(E)
        self.openness_buf.append(O)
        self.motion_buf.append(M)
        self.ts_buf.append(now)
        self.nose_vel_buf.append(v)
        self._ear_slope_buf.append(feats["ear"])

        # Blink detection
        is_closed = O < CLOSED_OPENNESS_THRESHOLD
        if is_closed and not self._in_blink:
            self._in_blink = True
            self._blink_start = now
        elif not is_closed and self._in_blink:
            self._in_blink = False
            if self._blink_start is not None:
                dur = now - self._blink_start
                self.blink_durations.append(dur)
                if dur >= BLINK_NORMAL_MAX:
                    self.long_blink_durations.append(dur)

        # Nod detection: downward spike in nose velocity
        if v > NOD_VELOCITY_THRESHOLD:
            self._nod_counter = NOD_LINGER_FRAMES
        if self._nod_counter > 0:
            self._nod_counter -= 1
            self._nod_flag = True
        else:
            self._nod_flag = False

    def compute_signals(self, now: float) -> dict:
        # --- PERCLOS ---
        n = len(self.openness_buf)
        if n == 0:
            perclos = 0.0
        else:
            closed_count = sum(1 for o in self.openness_buf if o < CLOSED_OPENNESS_THRESHOLD)
            perclos = closed_count / n

        # --- Long closure flag ---
        in_blink_dur = 0.0
        if self._in_blink and self._blink_start is not None:
            in_blink_dur = now - self._blink_start
        long_closure_flag = 1.0 if in_blink_dur > BLINK_NORMAL_MAX else 0.0
        # Also trigger if last blink was long
        if self.long_blink_durations:
            last_long = self.long_blink_durations[-1]
            long_closure_flag = max(long_closure_flag,
                                    min(1.0, (last_long - BLINK_NORMAL_MAX) / BLINK_LONG_MAX))

        # --- Motion entropy ---
        m_arr = np.array(self.motion_buf) if len(self.motion_buf) > 1 else np.array([0.0, 0.0])
        mean_m = float(np.mean(m_arr)) + 1e-8
        std_m = float(np.std(m_arr))
        entropy_approx = std_m / mean_m
        low_motion_flag = 1.0 if (mean_m < MOTION_LOW_THRESHOLD and
                                   entropy_approx < ENTROPY_LOW_THRESHOLD) else 0.0

        # --- EAR slope ---
        ear_arr = list(self._ear_slope_buf)
        if len(ear_arr) >= 4:
            diffs = np.diff(ear_arr)
            slope = float(np.mean(diffs))
        else:
            slope = 0.0
        sustained_ear_drop_flag = 1.0 if slope < EAR_SLOPE_FATIGUE else 0.0

        # --- Nod flag ---
        nod_flag = 1.0 if self._nod_flag else 0.0

        # --- Blink stats for display ---
        last_blink_dur = self.blink_durations[-1] if self.blink_durations else 0.0

        return {
            "perclos": perclos,
            "long_closure_flag": long_closure_flag,
            "low_motion_flag": low_motion_flag,
            "sustained_ear_drop_flag": sustained_ear_drop_flag,
            "nod_flag": nod_flag,
            "motion_level": mean_m,
            "last_blink_dur": last_blink_dur,
            "in_blink_dur": in_blink_dur,
        }


# ============================================================
# CONFIDENCE MODEL
# ============================================================

def compute_sleep_confidence(signals: dict) -> float:
    sc = (W_PERCLOS * signals["perclos"]
          + W_LONG_CLOSURE * signals["long_closure_flag"]
          + W_LOW_MOTION * signals["low_motion_flag"]
          + W_NOD * signals["nod_flag"]
          + W_EAR_DROP * signals["sustained_ear_drop_flag"])
    return float(np.clip(sc, 0.0, 1.0))


# ============================================================
# STATE MACHINE
# ============================================================

class StateMachine:
    def __init__(self):
        self.state = STATE_ALERT
        self._hold_since = None
        self._hold_value = None

    def update(self, smoothed: float, now: float) -> str:
        s = self.state

        if s == STATE_ALERT:
            if smoothed > TH_ALERT_TO_TIRED:
                if self._check_hold(smoothed, HOLD_ALERT_TO_TIRED, now):
                    self.state = STATE_TIRED
                    self._reset_hold()
            else:
                self._reset_hold()

        elif s == STATE_TIRED:
            if smoothed > TH_TIRED_TO_DROWSY:
                if self._check_hold(smoothed, HOLD_TIRED_TO_DROWSY, now):
                    self.state = STATE_DROWSY
                    self._reset_hold()
            elif smoothed < TH_ALERT_TO_TIRED:
                if self._check_hold(smoothed, HOLD_DROWSY_EXIT, now):
                    self.state = STATE_ALERT
                    self._reset_hold()
            else:
                self._reset_hold()

        elif s == STATE_DROWSY:
            if smoothed > TH_DROWSY_TO_MICROSLEEP:
                if self._check_hold(smoothed, HOLD_DROWSY_TO_MICROSLEEP, now):
                    self.state = STATE_MICROSLEEP
                    self._reset_hold()
            elif smoothed < TH_DROWSY_EXIT:
                if self._check_hold(smoothed, HOLD_DROWSY_EXIT, now):
                    self.state = STATE_ALERT
                    self._reset_hold()
            else:
                self._reset_hold()

        elif s == STATE_MICROSLEEP:
            if smoothed < TH_MICROSLEEP_EXIT:
                if self._check_hold(smoothed, HOLD_MICROSLEEP_EXIT, now):
                    self.state = STATE_DROWSY
                    self._reset_hold()
            else:
                self._reset_hold()

        return self.state

    def _check_hold(self, value, duration, now):
        if self._hold_since is None:
            self._hold_since = now
            self._hold_value = value
            return False
        return (now - self._hold_since) >= duration

    def _reset_hold(self):
        self._hold_since = None
        self._hold_value = None


# ============================================================
# VISUAL OUTPUT
# ============================================================

def draw_overlay(frame, state, feats, signals, confidence, smoothed, calib, calib_progress):
    h, w = frame.shape[:2]

    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color = STATE_COLORS.get(state, (255, 255, 255))
    y = 30

    def put(text, val_str, col=(220, 220, 220)):
        nonlocal y
        cv2.putText(frame, f"{text}: {val_str}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
        y += 24

    # State banner
    cv2.rectangle(frame, (0, 0), (300, 36), color, -1)
    cv2.putText(frame, f"STATE: {state}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
    y = 52

    if not calib.complete:
        pct = int(calib_progress * 100)
        put("CALIBRATING", f"{pct}%", (0, 200, 255))
        bar_w = int(280 * calib_progress)
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 10), (0, 200, 255), -1)
        return

    put("PERCLOS",    f"{signals['perclos']:.2f}", color)
    put("Confidence", f"{confidence:.2f}", color)
    put("Smoothed",   f"{smoothed:.2f}", color)
    put("EAR_norm",   f"{feats['E_norm']:.2f}")
    put("Open_norm",  f"{feats['O_norm']:.2f}")
    put("Motion_lvl", f"{signals['motion_level']:.4f}")
    put("Blink_dur",  f"{signals['last_blink_dur']:.2f}s")
    put("Nod",        str(bool(signals['nod_flag'])))
    put("EAR_drop",   str(bool(signals['sustained_ear_drop_flag'])))
    put("LowMotion",  str(bool(signals['low_motion_flag'])))

    # Confidence bar
    bar_x, bar_y = 10, y + 5
    bar_len = 280
    filled = int(bar_len * min(smoothed, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 12), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 12), color, -1)
    cv2.putText(frame, "SLEEP SCORE", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # ALARM banner
    if state == STATE_MICROSLEEP:
        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "! WAKE UP !  MICROSLEEP DETECTED",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    # Verify model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Download with:")
        print("  wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task -O face_landmarker.task")
        sys.exit(1)

    # MediaPipe setup
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Subsystems
    calib = Calibration()
    extractor = FeatureExtractor(w, h)
    temporal = TemporalModel()
    state_machine = StateMachine()
    alarm = AlarmController()

    smoothed_score = 0.0
    current_state = STATE_ALERT
    frame_idx = 0
    no_face_count = 0
    NO_FACE_RESET_FRAMES = TARGET_FPS * 3  # 3 sec without face resets calibration

    # Placeholder displays for when face not detected
    blank_feats = {"E_norm": 0.0, "O_norm": 0.0, "M_norm": 0.0,
                   "ear": 0.0, "opening": 0.0, "motion": 0.0, "nose_vel": 0.0}
    blank_signals = {"perclos": 0.0, "long_closure_flag": 0.0,
                     "low_motion_flag": 0.0, "sustained_ear_drop_flag": 0.0,
                     "nod_flag": 0.0, "motion_level": 0.0,
                     "last_blink_dur": 0.0, "in_blink_dur": 0.0}

    print("[INFO] Sleep detection system running. Press 'q' to quit.")
    print("[INFO] Calibrating for 3 seconds — keep eyes open and look forward.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed. Retrying...")
            time.sleep(0.05)
            continue

        now = time.monotonic()
        frame_idx += 1
        ts_ms = int(now * 1000)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            no_face_count += 1
            if no_face_count >= NO_FACE_RESET_FRAMES:
                # Reset calibration if face gone for too long
                calib.reset()
                extractor.prev_landmarks = None
                extractor.prev_nose_y = None
                no_face_count = 0
                print("[INFO] No face detected — recalibrating.")

            cv2.putText(frame, "No face detected", (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            draw_overlay(frame, current_state, blank_feats, blank_signals,
                         0.0, smoothed_score, calib, calib.progress(now))
            cv2.imshow("Sleep Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        no_face_count = 0
        landmarks = result.face_landmarks[0]

        # Feature extraction
        try:
            feats = extractor.extract(landmarks, calib)
        except Exception as ex:
            print(f"[WARN] Feature extraction error: {ex}")
            continue

        # Calibration phase
        calib_done = calib.update(
            feats["ear"], feats["opening"], feats["motion"], now)

        calib_progress = calib.progress(now)

        if not calib_done:
            draw_overlay(frame, STATE_ALERT, blank_feats, blank_signals,
                         0.0, 0.0, calib, calib_progress)
            cv2.imshow("Sleep Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Temporal modeling
        temporal.update(feats, now)
        signals = temporal.compute_signals(now)

        # Confidence model
        raw_confidence = compute_sleep_confidence(signals)

        # Temporal smoothing (EMA)
        smoothed_score = (ALPHA_EMA * raw_confidence
                          + (1.0 - ALPHA_EMA) * smoothed_score)

        # State machine
        current_state = state_machine.update(smoothed_score, now)

        # Alarm
        if current_state == STATE_MICROSLEEP:
            alarm.start()
        else:
            alarm.stop()

        # Draw
        draw_overlay(frame, current_state, feats, signals,
                     raw_confidence, smoothed_score, calib, 1.0)

        cv2.imshow("Sleep Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    alarm.stop()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()