"""
webcam_component.py  ·  InterviewAI — Body Language Monitor
============================================================
Cross-device compatible version with Metered.ca TURN server support.
TURN servers allow webcam streaming to work from any device/network.

Requires:
    pip install streamlit-webrtc av mediapipe ultralytics opencv-python-headless
"""

from __future__ import annotations

import os
import time
import threading
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import json
from streamlit_webrtc import WebRtcMode, RTCConfiguration, webrtc_streamer

# ═════════════════════════════════════════════════════════════════════════════
#  TURN / ICE server configuration  (Metered.ca)
# ═════════════════════════════════════════════════════════════════════════════

def _get_ice_servers() -> list:
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]


@st.cache_data(ttl=3600)
def get_ice_servers_cached() -> list:
    return _get_ice_servers()


def build_rtc_config() -> RTCConfiguration:
    servers = get_ice_servers_cached()
    return RTCConfiguration({"iceServers": servers})


# ═════════════════════════════════════════════════════════════════════════════
#  Landmark index constants
# ═════════════════════════════════════════════════════════════════════════════

LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33,  7,   163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]

L_EYE_LEFT_CORNER  = 263
L_EYE_RIGHT_CORNER = 362
R_EYE_LEFT_CORNER  = 133
R_EYE_RIGHT_CORNER = 33

NOSE_TIP = 4
CHIN     = 152
L_EAR    = 234
R_EAR    = 454

# ═════════════════════════════════════════════════════════════════════════════
#  Cheat-event type constants
# ═════════════════════════════════════════════════════════════════════════════

CHEAT_LOOKING_AWAY   = "LOOKING_AWAY"
CHEAT_HEAD_TURNED    = "HEAD_TURNED"
CHEAT_HEAD_DOWN      = "HEAD_DOWN"
CHEAT_MULTIPLE_FACES = "MULTIPLE_FACES"
CHEAT_EYES_CLOSED    = "EYES_CLOSED"
CHEAT_FREQUENT_AWAY  = "FREQUENT_AWAY"
CHEAT_PHONE_USAGE    = "PHONE_USAGE"
CHEAT_LONG_ABSENCE   = "LONG_ABSENCE"

# ═════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CheatEvent:
    timestamp:   float
    event_type:  str
    description: str
    severity:    str
    duration:    float = 0.0


@dataclass
class SessionMetrics:
    total_frames:          int   = 0
    frames_face_visible:   int   = 0
    eye_contact_frames:    int   = 0
    gaze_left_frames:      int   = 0
    gaze_right_frames:     int   = 0
    gaze_center_frames:    int   = 0
    gaze_up_frames:        int   = 0
    gaze_down_frames:      int   = 0
    head_away_frames:      int   = 0
    head_down_frames:      int   = 0
    blink_count:           int   = 0
    multiple_face_frames:  int   = 0
    phone_usage_frames:    int   = 0
    phone_events:          int   = 0
    face_absent_frames:    int   = 0
    nervousness_scores:    List[float] = field(default_factory=list)
    yaw_history:           List[float] = field(default_factory=list)
    pitch_history:         List[float] = field(default_factory=list)
    cheat_events:          List[CheatEvent] = field(default_factory=list)
    start_time:            float = field(default_factory=time.time)
    away_glance_count:     int   = 0
    currently_away:        bool  = False
    away_durations:        List[float] = field(default_factory=list)

    def eye_contact_pct(self) -> float:
        if not self.frames_face_visible:
            return 0.0
        return round(self.eye_contact_frames / self.frames_face_visible * 100, 1)

    def gaze_distribution(self) -> dict:
        t = self.frames_face_visible or 1
        return {
            "center": round(self.gaze_center_frames / t * 100, 1),
            "left":   round(self.gaze_left_frames   / t * 100, 1),
            "right":  round(self.gaze_right_frames  / t * 100, 1),
            "up":     round(self.gaze_up_frames     / t * 100, 1),
            "down":   round(self.gaze_down_frames   / t * 100, 1),
        }

    def avg_nervousness(self) -> float:
        if not self.nervousness_scores:
            return 0.0
        return round(sum(self.nervousness_scores) / len(self.nervousness_scores) * 10, 1)

    def head_stability(self) -> float:
        if len(self.yaw_history) < 2:
            return 10.0
        std_y = float(np.std(self.yaw_history))
        std_p = float(np.std(self.pitch_history))
        return round(max(0.0, 10.0 - (std_y + std_p) * 0.3), 1)

    def cheat_score(self) -> float:
        if not self.frames_face_visible:
            return 0.0
        t      = self.frames_face_visible
        tf     = self.total_frames or 1
        away   = (self.gaze_left_frames + self.gaze_right_frames) / t
        turn   = self.head_away_frames  / t
        down   = self.head_down_frames  / t
        multi  = self.multiple_face_frames / tf
        glance = min(self.away_glance_count * 0.5, 3.0)
        long_away = sum(1 for d in self.away_durations if d > 3.0)
        phone  = min(self.phone_events * 0.8, 3.0)
        absent = self.face_absent_frames / tf
        raw    = (away * 3.0 + turn * 2.5 + down * 2.0 + multi * 4.0
                  + glance * 0.3 + long_away * 0.5 + phone * 1.0 + absent * 2.5)
        return round(min(raw * 10, 10), 1)

    def blinks_per_minute(self) -> float:
        elapsed = max(time.time() - self.start_time, 1)
        return round(self.blink_count / elapsed * 60, 1)

    def integrity_score(self) -> float:
        return round(max(0.0, 10.0 - self.cheat_score()), 1)

    def summary(self) -> dict:
        cheat_s = self.cheat_score()
        return {
            "duration_seconds":      round(time.time() - self.start_time, 1),
            "face_visible_pct":      round(self.frames_face_visible / max(self.total_frames, 1) * 100, 1),
            "eye_contact_pct":       self.eye_contact_pct(),
            "gaze_distribution":     self.gaze_distribution(),
            "head_stability":        self.head_stability(),
            "avg_nervousness":       self.avg_nervousness(),
            "cheat_score":           cheat_s,
            "integrity_score":       self.integrity_score(),
            "blinks_per_minute":     self.blinks_per_minute(),
            "total_blinks":          self.blink_count,
            "away_glance_count":     self.away_glance_count,
            "multiple_face_events":  self.multiple_face_frames,
            "phone_events":          self.phone_events,
            "face_absent_frames":    self.face_absent_frames,
            "cheat_events": [
                {
                    "time":        round(e.timestamp - self.start_time, 1),
                    "type":        e.event_type,
                    "description": e.description,
                    "severity":    e.severity,
                    "duration":    round(e.duration, 1),
                }
                for e in self.cheat_events
            ],
            "scores": {
                "eye_contact":    round(self.eye_contact_pct() / 10, 1),
                "confidence":     round(max(0.0, 10.0 - self.avg_nervousness()), 1),
                "focus":          round(self.gaze_distribution()["center"] / 10, 1),
                "head_stability": self.head_stability(),
                "integrity":      self.integrity_score(),
                "cheat_risk":     cheat_s,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Core body-language analyser
# ═════════════════════════════════════════════════════════════════════════════

class BodyLanguageAnalyzer:
    def __init__(self):
        import mediapipe as mp
        self._mp_face_mesh      = mp.solutions.face_mesh
        self._mp_face_detection = mp.solutions.face_detection

        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_detector = self._mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

        self.session         = SessionMetrics()
        self._lock           = threading.Lock()
        self._blink_buf      = deque(maxlen=5)
        self._in_blink       = False
        self._away_start     : Optional[float] = None
        self._down_start     : Optional[float] = None
        self._multi_start    : Optional[float] = None
        self._phone_start    : Optional[float] = None
        self._absent_start   : Optional[float] = None
        self._gaze_history   = deque(maxlen=90)
        self._yaw_buf        = deque(maxlen=8)
        self._pitch_buf      = deque(maxlen=8)
        self._yolo           = None
        self._last_phone_det = 0.0
        self._init_yolo()

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
        except Exception:
            self._yolo = None

    def reset_session(self):
        with self._lock:
            self.session       = SessionMetrics()
            self._blink_buf.clear()
            self._in_blink     = False
            self._away_start   = None
            self._down_start   = None
            self._multi_start  = None
            self._phone_start  = None
            self._absent_start = None
            self._gaze_history.clear()
            self._yaw_buf.clear()
            self._pitch_buf.clear()

    @staticmethod
    def _ear(lm, eye_idx: list, w: int, h: int) -> float:
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in eye_idx])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return float((A + B) / (2.0 * C + 1e-6))

    @staticmethod
    def _iris_ratio(lm, iris_idx: list, corner_l: int, corner_r: int,
                    w: int, h: int) -> float:
        pts  = np.array([(lm[i].x * w, lm[i].y * h) for i in iris_idx])
        cx   = pts[:, 0].mean()
        lp   = np.array([lm[corner_l].x * w, lm[corner_l].y * h])
        rp   = np.array([lm[corner_r].x * w, lm[corner_r].y * h])
        span = float(np.linalg.norm(rp - lp)) + 1e-6
        return float((cx - lp[0]) / span)

    def _head_pose(self, lm, w: int, h: int) -> Tuple[float, float]:
        nose  = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
        chin  = np.array([lm[CHIN].x    * w, lm[CHIN].y    * h])
        l_ear = np.array([lm[L_EAR].x   * w, lm[L_EAR].y   * h])
        r_ear = np.array([lm[R_EAR].x   * w, lm[R_EAR].y   * h])
        l_eye = np.array([lm[33].x      * w, lm[33].y      * h])
        r_eye = np.array([lm[263].x     * w, lm[263].y     * h])

        ear_mid_x = (l_ear[0] + r_ear[0]) / 2.0
        ear_span  = abs(l_ear[0] - r_ear[0]) + 1e-6
        yaw       = float(np.degrees(np.arctan2(nose[0] - ear_mid_x, ear_span * 0.5)))
        yaw       = float(np.clip(yaw, -90, 90))

        eye_mid_y = (l_eye[1] + r_eye[1]) / 2.0
        face_h    = abs(chin[1] - eye_mid_y) + 1e-6
        pitch     = float(np.degrees(np.arctan2(nose[1] - eye_mid_y, face_h)) * 1.5)
        pitch     = float(np.clip(pitch, -60, 60))

        self._yaw_buf.append(yaw)
        self._pitch_buf.append(pitch)
        return float(np.mean(self._yaw_buf)), float(np.mean(self._pitch_buf))

    @staticmethod
    def _nervousness(yaw: float, pitch: float, ear: float, blink_rate: float) -> float:
        val = (min(abs(yaw) / 30, 1) * 0.35
               + min(abs(pitch) / 20, 1) * 0.20
               + min(max(blink_rate - 15, 0) / 25, 1) * 0.25
               + max(0.0, (0.25 - ear) / 0.25) * 0.20)
        return min(val, 1.0)

    def _log(self, etype: str, desc: str, severity: str, dur: float = 0.0):
        now = time.time()
        for e in reversed(self.session.cheat_events[-5:]):
            if e.event_type == etype and now - e.timestamp < 2.0:
                return
        self.session.cheat_events.append(CheatEvent(now, etype, desc, severity, dur))

    def _check_sustained(self, is_away: bool, is_down: bool, now: float):
        if is_away:
            if self._away_start is None:
                self._away_start = now
                if not self.session.currently_away:
                    self.session.away_glance_count += 1
                    self.session.currently_away     = True
            else:
                dur = now - self._away_start
                if dur > 2.0:
                    self._log(CHEAT_LOOKING_AWAY,
                              f"Sustained gaze away {dur:.1f}s",
                              "high" if dur > 4 else "medium", dur)
                if dur > 5.0:
                    self._log(CHEAT_FREQUENT_AWAY,
                              f"Extended off-screen {dur:.1f}s", "high", dur)
        else:
            if self._away_start:
                self.session.away_durations.append(now - self._away_start)
            self._away_start            = None
            self.session.currently_away = False

        if is_down:
            if self._down_start is None:
                self._down_start = now
            else:
                dur = now - self._down_start
                if dur > 2.5:
                    self._log(CHEAT_HEAD_DOWN, f"Head down {dur:.1f}s",
                              "high" if dur > 4 else "medium", dur)
        else:
            self._down_start = None

    def _check_phone_usage(self, pitch: float, gaze_dir: str, ear: float, now: float):
        is_phone_posture = pitch > 45 and gaze_dir == "down"
        if is_phone_posture:
            self.session.phone_usage_frames += 1
            if self._phone_start is None:
                self._phone_start = now
            else:
                dur = now - self._phone_start
                if dur > 4.0 and int(dur) % 6 == 0:
                    self.session.phone_events += 1
                    self._log(CHEAT_PHONE_USAGE,
                              f"Possible phone — head {pitch:.0f}° down + downward gaze {dur:.1f}s",
                              "high", dur)
        else:
            self._phone_start = None

    def _detect_phone_yolo(self, frame: np.ndarray) -> bool:
        if self._yolo is None:
            return False
        try:
            results = self._yolo(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label  = self._yolo.names[cls_id]
                    conf   = float(box.conf[0])
                    if label == "cell phone" and conf > 0.75:
                        now = time.time()
                        if now - self._last_phone_det > 5:
                            self._last_phone_det = now
                            self.session.phone_events += 1
                            self._log(CHEAT_PHONE_USAGE,
                                      "Phone detected by object-detection model", "high")
                        return True
        except Exception:
            pass
        return False

    def _check_face_absence(self, face_detected: bool, now: float):
        if not face_detected:
            self.session.face_absent_frames += 1
            if self._absent_start is None:
                self._absent_start = now
            else:
                dur = now - self._absent_start
                if dur > 3.0 and int(dur) % 4 == 0:
                    self._log(CHEAT_LONG_ABSENCE,
                              f"Face not visible for {dur:.1f}s", "high", dur)
        else:
            self._absent_start = None

    def _check_pattern(self):
        recent = list(self._gaze_history)[-60:]
        if len(recent) < 30:
            return
        away = sum(1 for g in recent if g in ("left", "right"))
        if away / len(recent) > 0.5:
            self._log(CHEAT_FREQUENT_AWAY,
                      f"Repeated off-screen glances: {away}/{len(recent)} frames", "medium")

    def _draw_hud(self, frame: np.ndarray, gaze_dir: str, head_dir: str,
                  yaw: float, pitch: float, nerv: float,
                  num_faces: int, cheat_s: float, w: int, h: int) -> np.ndarray:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (320, 190), (8, 8, 16), -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

        def col(v: float):
            if v >= 7: return (80, 255, 120)
            if v >= 5: return (80, 180, 255)
            return (60, 80, 255)

        gc  = (80, 255, 120) if gaze_dir == "center" else (60, 80, 255)
        fc  = (80, 255, 120) if num_faces == 1       else (60, 80, 255)
        pc  = (60, 80, 255)  if self.session.phone_events > 0 else (80, 255, 120)
        cs  = col(10.0 - cheat_s)
        ns  = col(round(10 - nerv * 10, 1))

        lines = [
            (f"GAZE  : {gaze_dir.upper()}",                    gc),
            (f"HEAD  : {head_dir.upper()}",                    (180, 180, 180)),
            (f"YAW   : {yaw:+.1f} deg",                        (180, 180, 180)),
            (f"PITCH : {pitch:+.1f} deg",                      (180, 180, 180)),
            (f"NERV  : {nerv * 100:.0f}%",                     ns),
            (f"FACES : {num_faces}",                            fc),
            (f"CHEAT : {cheat_s:.1f} / 10",                    cs),
            (f"PHONE : {self.session.phone_events} event(s)",  pc),
            (f"BLINK : {self.session.blinks_per_minute():.0f}/min", (180, 180, 180)),
            (f"GLNC  : {self.session.away_glance_count}",      (180, 180, 180)),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

        cx, cy = w - 58, 58
        cv2.circle(frame, (cx, cy), 30, (25, 25, 35), -1)
        cv2.circle(frame, (cx, cy), 30, (60, 60, 70), 1)
        ox, oy = {
            "center": (0, 0), "left": (-12, 0), "right": (12, 0),
            "up": (0, -12),   "down": (0, 12),  "none": (0, 0),
        }.get(gaze_dir, (0, 0))
        cv2.circle(frame, (cx + ox, cy + oy), 8, gc, -1)

        if num_faces > 1 or self.session.phone_events > 0:
            ab = frame.copy()
            cv2.rectangle(ab, (0, h - 50), (w, h), (160, 0, 0), -1)
            cv2.addWeighted(ab, 0.75, frame, 0.25, 0, frame)
            msg = ("  ALERT: MULTIPLE FACES"
                   if num_faces > 1 else "  ALERT: PHONE DETECTED")
            cv2.putText(frame, msg, (8, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, None]:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now  = time.time()

        with self._lock:
            self.session.total_frames += 1

            det       = self._face_detector.process(rgb)
            num_faces = len(det.detections) if det.detections else 0
            if num_faces > 1:
                self.session.multiple_face_frames += 1
                if self._multi_start is None:
                    self._multi_start = now
                else:
                    dur = now - self._multi_start
                    if dur > 1.5:
                        self._log(CHEAT_MULTIPLE_FACES,
                                  f"{num_faces} people detected in frame",
                                  "high" if dur > 3 else "medium", dur)
            else:
                self._multi_start = None

            if self.session.total_frames % 5 == 0:
                self._detect_phone_yolo(frame)

            mesh = self._face_mesh.process(rgb)
            self._check_face_absence(bool(mesh.multi_face_landmarks), now)

            if not mesh.multi_face_landmarks:
                frame = self._draw_hud(frame, "none", "none", 0, 0, 0,
                                       num_faces, self.session.cheat_score(), w, h)
                return frame, None

            lm = mesh.multi_face_landmarks[0].landmark
            self.session.frames_face_visible += 1

            ear_l = self._ear(lm, RIGHT_EYE_INDICES[:6], w, h)
            ear_r = self._ear(lm, LEFT_EYE_INDICES[:6],  w, h)
            ear   = (ear_l + ear_r) / 2.0

            if ear < 0.20 and not self._in_blink:
                self._in_blink = True
                self.session.blink_count += 1
            elif ear >= 0.20:
                self._in_blink = False
            if ear < 0.15:
                self._log(CHEAT_EYES_CLOSED, "Eyes closed", "low")

            blink_rate = self.session.blinks_per_minute()

            r_l = self._iris_ratio(lm, LEFT_IRIS,
                                   L_EYE_LEFT_CORNER, L_EYE_RIGHT_CORNER, w, h)
            r_r = self._iris_ratio(lm, RIGHT_IRIS,
                                   R_EYE_LEFT_CORNER, R_EYE_RIGHT_CORNER, w, h)
            ratio = (r_l + r_r) / 2.0

            if ratio < 0.37:
                gaze_dir = "left";   self.session.gaze_left_frames  += 1
            elif ratio > 0.63:
                gaze_dir = "right";  self.session.gaze_right_frames += 1
            else:
                dy = (np.mean([lm[i].y for i in LEFT_IRIS])
                      - np.mean([lm[i].y for i in LEFT_EYE_INDICES]))
                if dy < -0.007:
                    gaze_dir = "up";     self.session.gaze_up_frames    += 1
                elif dy > 0.007:
                    gaze_dir = "down";   self.session.gaze_down_frames  += 1
                else:
                    gaze_dir = "center"; self.session.gaze_center_frames += 1
                    self.session.eye_contact_frames += 1

            self._gaze_history.append(gaze_dir)

            yaw, pitch = self._head_pose(lm, w, h)
            self.session.yaw_history.append(yaw)
            self.session.pitch_history.append(pitch)

            self._check_phone_usage(pitch, gaze_dir, ear, now)

            is_away = abs(yaw) > 22
            is_down = pitch > 20
            if is_away:
                self.session.head_away_frames += 1
                head_dir = "LEFT" if yaw < 0 else "RIGHT"
                if abs(yaw) > 30:
                    self._log(CHEAT_HEAD_TURNED,
                              f"Head turned {head_dir} {abs(yaw):.0f}°",
                              "high" if abs(yaw) > 40 else "medium")
            elif is_down:
                self.session.head_down_frames += 1
                head_dir = "DOWN"
            elif pitch < -15:
                head_dir = "UP"
            elif yaw < -5:
                head_dir = "left"
            elif yaw > 5:
                head_dir = "right"
            else:
                head_dir = "center"

            self._check_sustained(
                is_away=(gaze_dir in ("left", "right") or is_away),
                is_down=(gaze_dir == "down" or is_down),
                now=now,
            )
            self._check_pattern()

            nerv = self._nervousness(yaw, pitch, ear, blink_rate)
            self.session.nervousness_scores.append(nerv)

            cheat_s = self.session.cheat_score()
            frame   = self._draw_hud(frame, gaze_dir, head_dir,
                                      yaw, pitch, nerv, num_faces, cheat_s, w, h)
            return frame, None

    def get_session_summary(self) -> dict:
        with self._lock:
            return self.session.summary()


# ═════════════════════════════════════════════════════════════════════════════
#  Singleton accessor
# ═════════════════════════════════════════════════════════════════════════════

_ANALYZER_INSTANCE: Optional[BodyLanguageAnalyzer] = None
_ANALYZER_LOCK = threading.Lock()


def get_analyzer() -> BodyLanguageAnalyzer:
    global _ANALYZER_INSTANCE
    with _ANALYZER_LOCK:
        if _ANALYZER_INSTANCE is None:
            _ANALYZER_INSTANCE = BodyLanguageAnalyzer()
        return _ANALYZER_INSTANCE


def reset_analyzer():
    global _ANALYZER_INSTANCE
    with _ANALYZER_LOCK:
        if _ANALYZER_INSTANCE is not None:
            _ANALYZER_INSTANCE.reset_session()
        else:
            _ANALYZER_INSTANCE = BodyLanguageAnalyzer()


# ═════════════════════════════════════════════════════════════════════════════
#  WebRTC video processor
# ═════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        analyzer = get_analyzer()
        processed, _ = analyzer.process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


# ═════════════════════════════════════════════════════════════════════════════
#  Score panel HTML builder
# ═════════════════════════════════════════════════════════════════════════════

def _score_color(score: float, invert: bool = False) -> str:
    v = (10.0 - score) if invert else score
    if v >= 7:  return "#c6ff4e"
    if v >= 5:  return "#ffb740"
    return "#ff5a5a"

def _score_bg(score: float, invert: bool = False) -> str:
    v = (10.0 - score) if invert else score
    if v >= 7:  return "rgba(198,255,78,.07)"
    if v >= 5:  return "rgba(255,183,64,.07)"
    return "rgba(255,90,90,.07)"

def _score_border(score: float, invert: bool = False) -> str:
    v = (10.0 - score) if invert else score
    if v >= 7:  return "rgba(198,255,78,.22)"
    if v >= 5:  return "rgba(255,183,64,.22)"
    return "rgba(255,90,90,.22)"

def _tile_html(label: str, value: str, suffix: str,
               color: str, bg: str, border: str) -> str:
    return f"""
      <div style="background:{bg};border:1px solid {border};border-radius:10px;
                  padding:10px 12px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;
                    color:rgba(240,236,228,.35);letter-spacing:.14em;
                    text-transform:uppercase;margin-bottom:5px;">{label}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;
                    font-weight:700;color:{color};line-height:1;">
          {value}<span style="font-size:11px;opacity:.5;">{suffix}</span>
        </div>
      </div>"""

def _build_scores_html(summary: dict) -> str:
    scores   = summary.get("scores", {})
    gaze     = summary.get("gaze_distribution", {})
    events   = summary.get("cheat_events", [])
    cheat_s  = summary.get("cheat_score", 0)
    phone_ev = summary.get("phone_events", 0)
    absent_f = summary.get("face_absent_frames", 0)
    ec       = scores.get("eye_contact",    5)
    cf       = scores.get("confidence",     5)
    fg       = scores.get("focus",          5)
    ig       = scores.get("integrity",      5)
    cr       = scores.get("cheat_risk",     0)
    hs       = scores.get("head_stability", 5)

    ph_color  = "#ff5a5a" if phone_ev > 0 else "#c6ff4e"
    ph_bg     = "rgba(255,90,90,.07)"  if phone_ev > 0 else "rgba(198,255,78,.07)"
    ph_border = "rgba(255,90,90,.22)"  if phone_ev > 0 else "rgba(198,255,78,.22)"
    ab_color  = "#ff5a5a" if absent_f > 45 else "#c6ff4e"
    ab_bg     = "rgba(255,90,90,.07)"  if absent_f > 45 else "rgba(198,255,78,.07)"
    ab_border = "rgba(255,90,90,.22)"  if absent_f > 45 else "rgba(198,255,78,.22)"
    cr_color  = _score_color(cr, invert=True)
    cr_bg     = _score_bg(cr,    invert=True)
    cr_border = _score_border(cr, invert=True)

    tiles = (
        _tile_html("Eye Contact",    f"{ec}",       "/10",     _score_color(ec), _score_bg(ec), _score_border(ec))
      + _tile_html("Confidence",     f"{cf}",       "/10",     _score_color(cf), _score_bg(cf), _score_border(cf))
      + _tile_html("Focus",          f"{fg}",       "/10",     _score_color(fg), _score_bg(fg), _score_border(fg))
      + _tile_html("Head Stability", f"{hs}",       "/10",     _score_color(hs), _score_bg(hs), _score_border(hs))
      + _tile_html("Integrity",      f"{ig}",       "/10",     _score_color(ig), _score_bg(ig), _score_border(ig))
      + _tile_html("Cheat Risk",     f"{cr}",       "/10",     cr_color, cr_bg, cr_border)
      + _tile_html("📱 Phone",        f"{phone_ev}", " events", ph_color, ph_bg, ph_border)
      + _tile_html("👁 Face Away",    f"{absent_f}", " frames", ab_color, ab_bg, ab_border)
    )

    focus_pct   = fg * 10
    focus_color = _score_color(fg)

    def gaze_pill(label, val, color, bg):
        return (f'<span style="background:{bg};border:1px solid {color};color:{color};'
                f'font-family:monospace;font-size:10px;padding:3px 9px;border-radius:12px;">'
                f'{label} {val}%</span>')

    gaze_pills = (
          gaze_pill("● Centre", gaze.get("center", 0), "rgba(198,255,78,.8)", "rgba(198,255,78,.08)")
        + gaze_pill("← Left",   gaze.get("left",   0), "rgba(255,90,90,.7)",  "rgba(255,90,90,.07)")
        + gaze_pill("→ Right",  gaze.get("right",  0), "rgba(255,90,90,.7)",  "rgba(255,90,90,.07)")
        + gaze_pill("↓ Down",   gaze.get("down",   0), "rgba(255,183,64,.7)", "rgba(255,183,64,.07)")
    )

    high_evs = [e for e in events if e["severity"] == "high"]
    med_evs  = [e for e in events if e["severity"] == "medium"]
    show_evs = (high_evs + med_evs)[-5:]
    events_html = ""
    if show_evs:
        rows = ""
        for e in show_evs:
            sev_c = "#ff5a5a" if e["severity"] == "high" else "#ffb740"
            sev_b = "rgba(255,90,90,.10)" if e["severity"] == "high" else "rgba(255,183,64,.10)"
            rows += f"""
            <div style="display:flex;align-items:flex-start;gap:8px;
                        padding:7px 0;border-bottom:1px solid rgba(240,236,228,.05);">
              <span style="background:{sev_b};color:{sev_c};font-family:monospace;
                           font-size:8px;padding:2px 7px;border-radius:5px;
                           white-space:nowrap;margin-top:1px;font-weight:700;">
                {e['severity'].upper()}
              </span>
              <div>
                <div style="font-size:11px;color:rgba(240,236,228,.65);line-height:1.4;">
                  {e['description']}
                </div>
                <div style="font-size:9px;color:rgba(240,236,228,.25);font-family:monospace;margin-top:2px;">
                  t+{e['time']}s{f" · {e['duration']}s" if e.get('duration') else ""}
                </div>
              </div>
            </div>"""
        events_html = f"""
        <div style="background:rgba(255,90,90,.05);border:1px solid rgba(255,90,90,.18);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
          <div style="font-family:monospace;font-size:9px;color:rgba(255,90,90,.6);
                      letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;">
            ⚠ Flagged Events
          </div>
          {rows}
        </div>"""

    footer_html = f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;padding-top:10px;
                border-top:1px solid rgba(240,236,228,.06);margin-top:10px;">
      <span style="font-family:monospace;font-size:10px;color:rgba(240,236,228,.4);">
        👁 {summary.get('eye_contact_pct', 0)}% eye contact
      </span>
      <span style="font-family:monospace;font-size:10px;color:rgba(240,236,228,.4);">
        😤 {summary.get('blinks_per_minute', 0)} blinks/min
      </span>
      <span style="font-family:monospace;font-size:10px;color:rgba(240,236,228,.4);">
        ↔ {summary.get('away_glance_count', 0)} off-screen glances
      </span>
      <span style="font-family:monospace;font-size:10px;color:rgba(240,236,228,.4);">
        ⏱ {summary.get('duration_seconds', 0)}s session
      </span>
    </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0a0a14; font-family:'IBM Plex Sans',system-ui,sans-serif;
          color:#f0ece4; padding:14px; }}
  @keyframes fade-in {{ from {{ opacity:0;transform:translateY(4px); }} to {{ opacity:1;transform:translateY(0); }} }}
  .panel {{ animation:fade-in .3s ease both; }}
</style></head><body>
<div class="panel">
  <div style="font-family:monospace;font-size:9px;color:rgba(90,176,255,.5);
              letter-spacing:.16em;text-transform:uppercase;margin-bottom:12px;">
    Live Scores · Body Language
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
    {tiles}
  </div>
  <div style="margin-bottom:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
      <span style="font-family:monospace;font-size:9px;color:rgba(240,236,228,.35);
                   letter-spacing:.12em;text-transform:uppercase;">Screen Focus</span>
      <span style="font-family:monospace;font-size:10px;color:{focus_color};font-weight:700;">{fg}/10</span>
    </div>
    <div style="background:rgba(240,236,228,.06);border-radius:6px;height:7px;overflow:hidden;">
      <div style="width:{focus_pct}%;background:{focus_color};border-radius:6px;height:7px;
                  box-shadow:0 0 8px {focus_color};transition:width .5s ease;"></div>
    </div>
  </div>
  <div style="margin-bottom:12px;">
    <div style="font-family:monospace;font-size:9px;color:rgba(240,236,228,.35);
                letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px;">Gaze Distribution</div>
    <div style="display:flex;gap:5px;flex-wrap:wrap;">{gaze_pills}</div>
  </div>
  {footer_html}
  {events_html}
</div></body></html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  TTS helpers
# ═════════════════════════════════════════════════════════════════════════════

def announce_question(question_number: int, question_text: str):
    _speak(f"Question {question_number}. {question_text}", rate=0.92)

def announce_summary(summary_text: str):
    _speak(summary_text[:2500], rate=0.88)

def _speak(text: str, rate: float = 0.92):
    safe = json.dumps(text)
    components.html(f"""<!DOCTYPE html><html><body><script>
    (function() {{
        var utterance = new SpeechSynthesisUtterance({safe});
        utterance.lang = 'en-US'; utterance.rate = {rate};
        utterance.pitch = 1.0; utterance.volume = 1.0;
        function pickVoice() {{
            var voices = window.speechSynthesis.getVoices();
            return voices.find(v => v.lang.startsWith('en') && /female|samantha|google us/i.test(v.name))
                || voices.find(v => v.lang.startsWith('en')) || null;
        }}
        function go() {{
            var v = pickVoice();
            if (v) utterance.voice = v;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }}
        if (window.speechSynthesis.getVoices().length > 0) {{ go(); }}
        else {{ window.speechSynthesis.onvoiceschanged = go; setTimeout(go, 600); }}
    }})();
    </script></body></html>""", height=0)


# ═════════════════════════════════════════════════════════════════════════════
#  Main render function  ← KEY CHANGE: @st.fragment + col_webcam param
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment
def render_webcam_monitor(col_webcam=None) -> None:
    container = col_webcam if col_webcam is not None else st.container()
    with container:
        st.markdown('<div class="section-label">Body Language</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#5ab0ff;
                      box-shadow:0 0 8px rgba(90,176,255,.8);
                      animation:pulse-dot 2s infinite;flex-shrink:0;"></div>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                       color:rgba(90,176,255,.7);letter-spacing:.18em;text-transform:uppercase;">
            Live Monitor · Integrity Analysis
          </span>
        </div>
        <style>
          @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.7);} }
          div[data-testid="stCustomComponentV1"] > iframe {
            width:100% !important; height:420px !important;
            border-radius:14px !important;
            border:1px solid rgba(90,176,255,.2) !important;
            background:#06060c !important;
          }
        </style>
        """, unsafe_allow_html=True)

        rtc_config = build_rtc_config()

        ctx = webrtc_streamer(
            key="bl_monitor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {
                    "width":     {"ideal": 640},
                    "height":    {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
            desired_playing_state=True,
            translations={
                "start": "▶  Start Camera  (stays on for whole session)",
                "stop":  "■  Stop Camera",
            },
        )

        if not ctx.state.playing:
            components.html("""
            <!DOCTYPE html><html><body style="margin:0;background:#0e0e18;">
            <div style="border:1px dashed rgba(90,176,255,.2);border-radius:14px;
                        padding:40px 20px;text-align:center;font-family:monospace;">
              <div style="font-size:36px;margin-bottom:12px;">📹</div>
              <div style="font-size:11px;color:rgba(90,176,255,.5);line-height:1.9;">
                Click <strong style="color:#5ab0ff;">Start Camera</strong> above<br/>
                <span style="font-size:10px;color:rgba(90,176,255,.3);">
                  Only needed once — persists across all questions
                </span>
              </div>
            </div>
            </body></html>
            """, height=160)
            return

        summary = get_final_bl_summary()
        scores_html = _build_scores_html(summary)
        components.html(scores_html, height=800, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
#  Public helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_final_bl_summary() -> dict:
    analyzer = get_analyzer()
    summary  = analyzer.get_session_summary()
    scores   = summary.get("scores", {})
    gaze     = summary.get("gaze_distribution", {})
    events   = summary.get("cheat_events", [])
    cheat_s  = summary.get("cheat_score", 0)
    phone_ev = summary.get("phone_events", 0)
    absent_f = summary.get("face_absent_frames", 0)

    high_events = [e for e in events if e["severity"] == "high"]
    observations = [
        f"Eye contact maintained {summary.get('eye_contact_pct', 0)}% of session",
        f"Gaze on screen (centre): {gaze.get('center', 0)}%",
        f"Off-screen glances: {summary.get('away_glance_count', 0)} times",
        f"Blink rate: {summary.get('blinks_per_minute', 0)}/min (normal 15–20)",
    ]
    if phone_ev > 0:
        observations.append(f"Phone detected: {phone_ev} time(s)")
    if absent_f > 45:
        observations.append(f"Face absent from frame: {absent_f} frames (~{round(absent_f/15,1)}s)")
    if high_events:
        observations.append(f"High-severity integrity events: {len(high_events)}")
    if summary.get("multiple_face_events", 0) > 0:
        observations.append(f"Multiple faces detected: {summary['multiple_face_events']} frames")

    return {
        "eye_contact":      scores.get("eye_contact",    5),
        "confidence":       scores.get("confidence",     5),
        "posture":          scores.get("head_stability", 5),
        "overall_presence": round(sum([
            scores.get("eye_contact",    5),
            scores.get("confidence",     5),
            scores.get("focus",          5),
            scores.get("head_stability", 5),
            scores.get("integrity",      5),
        ]) / 5, 1),
        "cheat_score":           cheat_s,
        "integrity_score":       scores.get("integrity", 5),
        "phone_events":          phone_ev,
        "face_absent_frames":    absent_f,
        "multiple_face_events":  summary.get("multiple_face_events", 0),
        "expression": (
            f"Eye contact {summary.get('eye_contact_pct', 0)}% · "
            f"Cheat risk {cheat_s}/10"
        ),
        "observations": observations,
        "tip":          _derive_tip(summary),
        "cheat_events": events,
        "full_summary": summary,
    }


def _derive_tip(summary: dict) -> str:
    scores   = summary.get("scores", {})
    gaze     = summary.get("gaze_distribution", {})
    events   = summary.get("cheat_events", [])
    high_evs = [e for e in events if e["severity"] == "high"]
    phone_ev = summary.get("phone_events", 0)
    absent_f = summary.get("face_absent_frames", 0)

    if summary.get("multiple_face_events", 0) > 10:
        return "Another person was detected — ensure you are alone for the interview."
    if phone_ev > 0:
        return f"Phone detected {phone_ev} time(s) — put it away and out of sight."
    if absent_f > 45:
        return "Stay centred in frame — your face disappeared for extended periods."
    if high_evs:
        return f"Suspicious behaviour flagged {len(high_evs)} time(s) — avoid looking away."
    if scores.get("eye_contact", 5) < 5:
        return "Maintain more eye contact with the camera — aim for 70%+ of the time."
    if scores.get("integrity", 5) < 5:
        off = round(gaze.get("left", 0) + gaze.get("right", 0), 1)
        return f"You looked off-screen {off}% of the time — stay focused on the camera."
    if scores.get("confidence", 5) < 5:
        return "Slow down, breathe, and steady your head movements for a calmer presence."
    return "Great presence! Keep maintaining eye contact and an upright, relaxed posture."
