"""9:16 layout choice (facecam_top | center_crop | blur_fit) and ffmpeg filtergraphs.
Never invents a facecam: it needs a face seen at a consistent spot in most sampled frames."""
from __future__ import annotations
import numpy as np

N_SAMPLES = 6
FACE_MIN_HITS = 4          # of 6 frames to accept a facecam
FACE_MAYBE_HITS = 2        # 2-3 hits = ambiguous -> no facecam + layout_review
CENTER_RATIO_CUT = 0.55    # share of motion inside the center 9:16 band to use center crop
CENTER_RATIO_FUZZ = 0.10   # within this of the cut -> layout_review
MAX_FACE_AREA = 0.45


def sample_times(duration: float, n: int = N_SAMPLES) -> list[float]:
    return [duration * (i + 0.5) / n for i in range(n)]


def read_frames(path, times: list[float], gap: float = 0.25, max_w: int = 640):
    """[(gray_a, gray_b)] near each time (b is `gap` later) + (src_w, src_h)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = []

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, f = cap.read()
        if not ok:
            return None
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sc = max_w / g.shape[1]
        return cv2.resize(g, (max_w, int(g.shape[0] * sc))) if sc < 1 else g

    for t in times:
        a, b = grab(t), grab(t + gap)
        if a is not None and b is not None and a.shape == b.shape:
            out.append((a, b))
    cap.release()
    return out, (src_w, src_h)


def haar_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    import cv2
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    mn = max(24, gray.shape[1] // 25)
    return [tuple(int(v) for v in f) for f in cc.detectMultiScale(gray, 1.1, 5, minSize=(mn, mn))]


def facecam_from_faces(face_lists: list[list[tuple]], shape: tuple[int, int], tol: float = 0.08):
    """Most consistent face position across frames -> (hits, padded rect in frame px) or (hits, None)."""
    h, w = shape
    best = []
    for fl in face_lists:
        if not fl:
            continue
        x, y, fw, fh = max(fl, key=lambda b: b[2] * b[3])
        best.append((x + fw / 2, y + fh / 2, fw, fh))
    if not best:
        return 0, None
    cx, cy = np.median([b[0] for b in best]), np.median([b[1] for b in best])
    close = [b for b in best if abs(b[0] - cx) <= tol * w and abs(b[1] - cy) <= tol * h]
    hits = len(close)
    if hits < FACE_MAYBE_HITS:
        return hits, None
    fw, fh = np.median([b[2] for b in close]), np.median([b[3] for b in close])
    rw, rh = fw * 2.0, fh * 2.0                              # facecam frame is bigger than the face
    x0, y0 = max(0, cx - rw / 2), max(0, cy - rh / 2)
    x1, y1 = min(w, cx + rw / 2), min(h, cy + rh / 2)
    m = 0.12                                                   # snap to the edge when near it (corner cams)
    if x0 < m * w: x0 = 0
    if y0 < m * h: y0 = 0
    if x1 > (1 - m) * w: x1 = w
    if y1 > (1 - m) * h: y1 = h
    return hits, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def motion_center_ratio(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Share of frame-difference energy inside the centered 9:16 band (0.5625 of width at 16:9)."""
    if not pairs:
        return 0.5
    h, w = pairs[0][0].shape
    band_w = min(w, h * 9 / 16)
    x0, x1 = int((w - band_w) / 2), int((w + band_w) / 2)
    cols = sum(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=0) for a, b in pairs)
    tot = float(cols.sum())
    return float(cols[x0:x1].sum() / tot) if tot > 0 else 0.5


def choose_layout(path, duration: float, detect=haar_faces, offset: float = 0.0) -> dict:
    pairs, (sw, sh) = read_frames(path, [offset + t for t in sample_times(duration)])
    if not pairs or not sw or not sh:
        return {"kind": "blur_fit", "review": True, "rect": None, "src": (sw, sh), "info": "no frames read"}
    fh, fw = pairs[0][0].shape
    hits, rect = facecam_from_faces([detect(a) for a, _ in pairs], (fh, fw))
    ratio = motion_center_ratio(pairs)
    info = {"face_hits": hits, "center_ratio": round(ratio, 2)}
    review = False
    if rect and hits >= FACE_MIN_HITS and rect[2] * rect[3] <= MAX_FACE_AREA * fw * fh:
        sx = sw / fw
        r = tuple(int(v * sx) // 2 * 2 for v in rect)
        return {"kind": "facecam_top", "rect": r, "review": hits < len(pairs) - 1, "src": (sw, sh), "info": info}
    if hits >= FACE_MAYBE_HITS:
        review = True                                          # saw something face-like, not sure
    if abs(ratio - CENTER_RATIO_CUT) <= CENTER_RATIO_FUZZ:
        review = True
    kind = "center_crop" if ratio >= CENTER_RATIO_CUT else "blur_fit"
    return {"kind": kind, "rect": None, "review": review, "src": (sw, sh), "info": info}


def choose_layout_at(path, offset: float, duration: float, detect=haar_faces) -> dict:
    """Like choose_layout, but sampling only the clip's span inside a padded section file."""
    return choose_layout(path, duration, detect, offset)


def top_height(H: int, frac: float) -> int:
    return int(round(H * frac / 2)) * 2


def build_filter(layout: dict, W: int, H: int, top_frac: float = 0.34, ass: str = "captions.ass") -> str:
    k = layout["kind"]
    tail = f"fps=30,ass={ass}[v]"
    if k == "center_crop":
        return f"[0:v]crop='min(iw,ih*{W}/{H})':ih,scale={W}:{H},{tail}"
    if k == "blur_fit":
        return (f"[0:v]split=2[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},boxblur=25:3[bg];[b]scale={W}:-2[fg];"
                f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,{tail}")
    if k == "facecam_top":
        x, y, w, h = layout["rect"]
        Ht = top_height(H, top_frac)
        Hb = H - Ht
        return (f"[0:v]split=2[f][g];[f]crop={w}:{h}:{x}:{y},scale={W}:{Ht}:force_original_aspect_ratio=increase,"
                f"crop={W}:{Ht}[top];[g]crop='min(iw,ih*{W}/{Hb})':ih,scale={W}:{Hb}[bot];"
                f"[top][bot]vstack,{tail}")
    raise ValueError(f"unknown layout {k}")
