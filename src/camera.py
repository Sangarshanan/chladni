"""Background camera capture thread + GestureInput, a Node exposing
hand-tracking signals as live Signal plus CameraOverlay
to render the live annotated camera feed directly on the canvas.
"""

import math
from pathlib import Path
import threading
import time
import urllib.request
import numpy as np

from .core import Node
from .signals import Signal
from .visuals import Geometry

import moderngl
import cv2 as _cv2
import mediapipe as _mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm connections
]

_WRIST       = 0
_THUMB_IP    = 3
_THUMB_TIP   = 4
_INDEX_TIP   = 8
_MIDDLE_MCP  = 9
_PINKY_MCP   = 17
_FINGER_MCPS = [5, 9, 13, 17]   # index, middle, ring, pinky
_FINGER_TIPS = [8, 12, 16, 20]  # index, middle, ring, pinky

# Target capture dimensions — small enough to be cheap, big enough for hands.
_FRAME_W = 320
_FRAME_H = 240
_TARGET_FPS = 30

_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def to_pixel(x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
    """Convert normalized landmark coordinates to pixel coordinates."""
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)


def draw_hand_landmarks(
    image_bgr: np.ndarray,
    hand_landmarks_list,
    connections=HAND_CONNECTIONS,
    draw_points=True,
    draw_connections=True,
    point_radius=3,
    point_thickness=-1,
    line_thickness=2,
):
    """Draw hand landmarks and skeleton connections on an image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    for hand_landmarks in hand_landmarks_list:
        pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]

        if draw_connections:
            for a, b in connections:
                _cv2.line(annotated, pts[a], pts[b], (0, 255, 0), line_thickness)

        if draw_points:
            for (x, y) in pts:
                _cv2.circle(annotated, (x, y), point_radius, (0, 0, 255), point_thickness)

    return annotated


def _draw_hud(img: np.ndarray, gestures: dict | None):
    """Draw gesture metrics HUD and border directly on the camera frame."""
    if _cv2 is None:
        return
    h, w = img.shape[:2]

    # Semi-transparent dark banner at bottom for metrics
    overlay = img.copy()
    _cv2.rectangle(overlay, (0, h - 50), (w, h), (10, 10, 15), -1)
    _cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    if gestures and gestures.get("hand_present", 0.0) > 0.5:
        p_val = gestures["pinch"]
        o_val = gestures["openness"]
        f_val = int(round(gestures["fingers"] * 5))
        hx, hy = gestures["hand_x"], gestures["hand_y"]

        _cv2.putText(
            img, f"Pinch: {p_val:.2f}", (10, h - 30),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, _cv2.LINE_AA,
        )
        _cv2.putText(
            img, f"Open: {o_val:.2f}", (115, h - 30),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, _cv2.LINE_AA,
        )
        _cv2.putText(
            img, f"Fingers: {f_val}/5", (215, h - 30),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, _cv2.LINE_AA,
        )
        _cv2.putText(
            img, f"Pos: ({hx:+.2f}, {hy:+.2f})", (10, h - 12),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, _cv2.LINE_AA,
        )


def _ensure_model(custom_path=None) -> Path | None:
    """Find or download the MediaPipe hand_landmarker.task model."""
    if custom_path is not None:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p

    candidates = [
        Path.cwd() / _MODEL_FILENAME,
        Path(__file__).parent / _MODEL_FILENAME,
        Path.home() / ".cache" / "chladni" / _MODEL_FILENAME,
    ]
    for c in candidates:
        if c.exists():
            return c

    target = Path.home() / ".cache" / "chladni" / _MODEL_FILENAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[chladni] Downloading {_MODEL_FILENAME} to {target}...")
        urllib.request.urlretrieve(_MODEL_URL, target)
        print("[chladni] Model download complete.")
        return target
    except Exception as e:
        print(
            f"[chladni] Warning: Failed to download hand landmarker model: {e}\n"
            f"Please download it manually from:\n{_MODEL_URL}\n"
            f"and place it at '{target}' or in the working directory."
        )
        return None


def _lm(landmarks, idx):
    """Return (x, y) of a MediaPipe landmark as floats in [0, 1]."""
    lm = landmarks.landmark[idx] if hasattr(landmarks, "landmark") else landmarks[idx]
    return lm.x, lm.y


def _dist(landmarks, a, b):
    """Euclidean distance between two landmarks (in normalised image coords)."""
    ax, ay = _lm(landmarks, a)
    bx, by = _lm(landmarks, b)
    return math.hypot(ax - bx, ay - by)


def _extended_fingers(landmarks):
    """Count extended fingers (0-5)."""
    count = 0
    for tip, mcp in zip(_FINGER_TIPS, _FINGER_MCPS):
        # In image coords y increases downwards, so 'raised' means lower y.
        if _lm(landmarks, tip)[1] < _lm(landmarks, mcp)[1]:
            count += 1
    # Thumb: distance from thumb tip to pinky MCP compared to IP distance.
    thumb_tip_dist = _dist(landmarks, _THUMB_TIP, _PINKY_MCP)
    thumb_ip_dist = _dist(landmarks, _THUMB_IP, _PINKY_MCP)
    if thumb_tip_dist > thumb_ip_dist:
        count += 1
    return count


def _compute_gestures(landmarks):
    """Return a dict of scale-invariant normalised gesture values from a hand's landmarks."""
    # Position: wrist in image coords -> NDC (-1..1, with y flipped).
    wx, wy = _lm(landmarks, _WRIST)
    hand_x = (wx - 0.5) * 2.0         # 0->-1, 1->+1 (left=-1, right=+1)
    hand_y = -(wy - 0.5) * 2.0        # 0->+1, 1->-1 (top=+1, bottom=-1)

    # Reference palm size (wrist to middle finger MCP) for depth/scale invariance.
    palm_size = max(_dist(landmarks, _WRIST, _MIDDLE_MCP), 1e-4)

    # Pinch: thumb-to-index tip distance normalised by palm size.
    raw_pinch = _dist(landmarks, _THUMB_TIP, _INDEX_TIP) / palm_size
    pinch = (raw_pinch - 0.15) / 1.2

    # Extended fingers (0-5) -> 0..1.
    n_fingers = _extended_fingers(landmarks)
    fingers = n_fingers / 5.0

    # Openness: mean distance from each fingertip to wrist, normalised by palm size.
    tip_dists = [_dist(landmarks, tip, _WRIST) / palm_size for tip in _FINGER_TIPS]
    tip_dists.append(_dist(landmarks, _THUMB_TIP, _WRIST) / palm_size)
    mean_ratio = sum(tip_dists) / len(tip_dists)
    openness = (mean_ratio - 0.75) / 1.15

    return {
        "hand_x":       float(max(-1.0, min(1.0, hand_x))),
        "hand_y":       float(max(-1.0, min(1.0, hand_y))),
        "pinch":        float(max(0.0, min(1.0, pinch))),
        "fingers":      float(max(0.0, min(1.0, fingers))),
        "openness":     float(max(0.0, min(1.0, openness))),
        "hand_present": 1.0,
    }


_ABSENT = {
    "hand_x":       0.0,
    "hand_y":       0.0,
    "pinch":        0.0,
    "fingers":      0.0,
    "openness":     0.0,
    "hand_present": 0.0,
}


class _CameraStream:
    """Capture frames and running MediaPipe Tasks HandLandmarker."""

    _instance = None

    def __init__(self, device=0, model_path=None):
        self._lock = threading.Lock()
        self._gestures = dict(_ABSENT)
        self._latest_annotated = None
        self._running = True

        model_file = _ensure_model(model_path)
        if model_file is None or not Path(model_file).exists():
            raise RuntimeError(
                f"MediaPipe task model '{_MODEL_FILENAME}' not found or could not be loaded."
            )

        base_options = _mp_python.BaseOptions(model_asset_path=str(model_file))
        options = _mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=_mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = _mp_vision.HandLandmarker.create_from_options(options)

        cap = _cv2.VideoCapture(device)
        cap.set(_cv2.CAP_PROP_FRAME_WIDTH, _FRAME_W)
        cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, _FRAME_H)
        cap.set(_cv2.CAP_PROP_FPS, _TARGET_FPS)
        if not cap.isOpened():
            self._landmarker.close()
            raise RuntimeError(f"Could not open camera device {device}")

        self._thread = threading.Thread(
            target=self._loop,
            args=(cap, self._landmarker),
            daemon=True,
            name="chladni-camera",
        )
        self._thread.start()

    def _loop(self, cap, landmarker):
        last_ts_ms = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = _cv2.flip(frame, 1)
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode requires strictly monotonically increasing timestamps in ms
            ts_ms = max(last_ts_ms + 1, int(time.perf_counter() * 1000))
            last_ts_ms = ts_ms

            try:
                result = landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                continue

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                lms = result.hand_landmarks[0]
                gestures = _compute_gestures(lms)
                annotated = draw_hand_landmarks(frame, [lms])
                fh, fw = annotated.shape[:2]

                # Highlight active pinch midpoint using actual frame dimensions
                if gestures["pinch"] < 0.25:
                    p_thumb = to_pixel(_lm(lms, _THUMB_TIP)[0], _lm(lms, _THUMB_TIP)[1], fw, fh)
                    p_index = to_pixel(_lm(lms, _INDEX_TIP)[0], _lm(lms, _INDEX_TIP)[1], fw, fh)
                    mid_pt = ((p_thumb[0] + p_index[0]) // 2, (p_thumb[1] + p_index[1]) // 2)
                    _cv2.circle(annotated, mid_pt, 7, (0, 255, 255), -1)
                    _cv2.circle(annotated, mid_pt, 10, (0, 200, 255), 1)
                _draw_hud(annotated, gestures)
            else:
                gestures = dict(_ABSENT)
                annotated = frame.copy()
                _draw_hud(annotated, None)

            annotated_rgb = _cv2.cvtColor(annotated, _cv2.COLOR_BGR2RGB)

            with self._lock:
                self._gestures = gestures
                self._latest_annotated = annotated_rgb

        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass

    def read(self):
        with self._lock:
            return dict(self._gestures)

    def read_annotated_frame(self):
        with self._lock:
            return self._latest_annotated.copy() if self._latest_annotated is not None else None

    def stop(self):
        self._running = False

    @classmethod
    def instance(cls, device=0, model_path=None):
        if cls._instance is None:
            cls._instance = cls(device=device, model_path=model_path)
        return cls._instance


# Signal and Node

class _GestureSignal(Signal):
    """A live Signal reading one gesture value off a GestureInput node."""

    def __init__(self, owner, key):
        self._owner = owner
        self._key = key

    def value(self):
        return self._owner._values[self._key]


class GestureInput(Node):
    """Reads hand-tracking data from camera each frame.

    Exposed signals (all smoothed):
        .hand_x       -- hand x position, -1..1 in NDC
        .hand_y       -- hand y position, -1..1 in NDC
        .pinch        -- thumb-index pinch, 0 (closed) .. 1 (open)
        .fingers      -- fraction of fingers raised, 0..1
        .openness     -- hand openness, 0 (fist) .. 1 (open palm)
        .hand_present -- 1.0 when a hand is detected, else 0.0

    Example:

        gesture = GestureInput()
        circle.x = gesture.hand_x
        circle.radius = gesture.pinch.map(0.1, 0.5)
        particles.turbulence = gesture.openness.map(0.2, 3.0)
        gesture.overlay >> output
    """

    def __init__(self, device=0, smoothing=0.08, model_path=None, show_window=False):
        super().__init__()
        self.smoothing = smoothing
        self.model_path = model_path
        self.show_window = show_window
        self._values = dict(_ABSENT)
        self._overlay_node = None

        self._stream = _CameraStream.instance(device=device, model_path=model_path)

        self.hand_x       = _GestureSignal(self, "hand_x")
        self.hand_y       = _GestureSignal(self, "hand_y")
        self.pinch        = _GestureSignal(self, "pinch")
        self.fingers      = _GestureSignal(self, "fingers")
        self.openness     = _GestureSignal(self, "openness")
        self.hand_present = _GestureSignal(self, "hand_present")

    @property
    def overlay(self):
        """Return a CameraOverlay node showing this gesture feed in a corner PiP."""
        if self._overlay_node is None:
            self._overlay_node = CameraOverlay(self)
        return self._overlay_node

    def get_annotated_frame(self):
        """Return the latest RGB camera frame annotated with landmarks and HUD."""
        return self._stream.read_annotated_frame()

    def update(self, dt):
        raw = self._stream.read()
        alpha = 1.0 - math.exp(-dt / max(self.smoothing, 1e-6))

        # Always smooth hand_present toward current detection state
        self._values["hand_present"] += (raw["hand_present"] - self._values["hand_present"]) * alpha

        # Only update spatial/gesture parameters when a hand is visible; hold last values otherwise
        if raw["hand_present"] > 0.5:
            for key in ("hand_x", "hand_y", "pinch", "fingers", "openness"):
                self._values[key] += (raw[key] - self._values[key]) * alpha

        if self.show_window and _cv2 is not None:
            frame = self._stream.read_annotated_frame()
            if frame is not None:
                bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
                _cv2.imshow("chladni - Camera Gestures", bgr)
                _cv2.waitKey(1)


class CameraOverlay(Geometry):
    """Picture-in-picture camera feed with live hand-tracking skeleton and gesture HUD.

    Connect with >> to route into Output:
        camera = CameraOverlay(gesture)
        camera >> output
        # or simply:
        gesture.overlay >> output
    """

    def __init__(
        self, gesture: GestureInput,
        alpha = 0.3,
        x=-1,
        y=-1,
        width=2,
        height=2,
    ):
        super().__init__()
        self.gesture = gesture
        self.alpha = alpha
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self._tex = None
        self._prog = None
        self._vbo = None
        self._vao = None
        self._bounds = None

    def render(self, ctx):
        """Render the camera frame and HUD into the ModernGL viewport."""
        frame = self.gesture.get_annotated_frame()
        if frame is None:
            return

        h, w = frame.shape[:2]
        if self._prog is None:
            self._prog = ctx.program(
                vertex_shader="""#version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = in_uv;
                }""",
                fragment_shader="""#version 330
                uniform sampler2D tex;
                uniform float alpha;
                in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    f_color = vec4(texture(tex, v_uv).rgb * alpha, 1);
                }""",
            )
            self._vbo = ctx.buffer(reserve=24 * 4, dynamic=True)
            self._vao = ctx.vertex_array(self._prog, [(self._vbo, "2f 2f", "in_pos", "in_uv")])

        if self._tex is None or self._tex.size != (w, h):
            if self._tex is not None:
                self._tex.release()
            self._tex = ctx.texture((w, h), 3)

        bounds = (float(self.x), float(self.y), float(self.width), float(self.height))
        if self._bounds != bounds:
            self._bounds = bounds
            x0, y0, bw, bh = bounds
            x1, y1 = x0 + bw, y0 + bh
            # (0, 1) top UV to (1, 0) bottom UV to match top-down image coords
            quad = np.array([
                x0, y0, 0.0, 1.0,
                x1, y0, 1.0, 1.0,
                x1, y1, 1.0, 0.0,
                x0, y0, 0.0, 1.0,
                x1, y1, 1.0, 0.0,
                x0, y1, 0.0, 0.0,
            ], dtype="f4")
            self._vbo.write(quad.tobytes())

        self._tex.write(frame.tobytes())
        self._tex.use(0)
        self._prog["tex"].value = 0
        self._prog["alpha"].value = self.alpha
        self._vao.render(moderngl.TRIANGLES)

    def close(self):
        """Release ModernGL GPU resources when pipeline hot-reloads."""
        for resource in (self._vao, self._vbo, self._tex, self._prog):
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
        self._vao = self._vbo = self._tex = self._prog = None

    def vertices(self):
        # Fallback when running without custom render
        return np.zeros((0, 2), dtype="f4"), np.zeros((0, 3), dtype="f4"), "points"