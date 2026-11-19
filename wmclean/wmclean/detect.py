"""Static watermark detection.

Idea: a static overlay has edges in EVERY frame at the SAME pixels. Background
edges move around. So take the 25th percentile of gradient magnitude across
frames sampled over the whole video; only static overlay edges survive.
"""
import json
import subprocess

import cv2
import numpy as np


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", path])
    j = json.loads(out)
    s, f = j["streams"][0], j["format"]
    num, den = s["avg_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den or 1), "duration": float(f["duration"])}


def sample_gray_frames(path, n=32):
    info = probe(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for t in np.linspace(info["duration"] * 0.03, info["duration"] * 0.97, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 8:
        raise RuntimeError("Could not sample enough frames from the video.")
    return frames, info


def _rois(w, h, roi):
    if roi:
        x, y, rw, rh = roi
        return [(x, y, rw, rh)]
    cw, ch = int(w * 0.38), int(h * 0.32)
    return [(0, 0, cw, ch), (w - cw, 0, cw, ch),
            (0, h - ch, cw, ch), (w - cw, h - ch, cw, ch)]


def detect_mask(path, roi=None, n_frames=32, grow=7):
    """Returns (mask uint8 0/255 full-frame, info dict, report str)."""
    frames, info = sample_gray_frames(path, n_frames)
    h, w = frames[0].shape
    mask = np.zeros((h, w), np.uint8)
    notes = []
    for (x, y, rw, rh) in _rois(w, h, roi):
        grads = []
        for g in frames:
            c = cv2.GaussianBlur(g[y:y + rh, x:x + rw], (3, 3), 0)
            gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
            grads.append(cv2.magnitude(gx, gy))
        score = np.percentile(np.stack(grads), 25, axis=0)
        binary = (score > 40).astype(np.uint8) * 255
        frac = binary.mean() / 255
        if frac > 0.25:
            notes.append(f"ROI@{x},{y}: {frac:.0%} 'static' - static scene/letterbox, skipped")
            continue
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
        keep = np.zeros_like(binary)
        for i in range(1, n_lab):
            cx, cy, cw, ch, area = stats[i]
            if area < 60:
                continue
            if (cw >= 0.9 * rw and ch <= 4) or (ch >= 0.9 * rh and cw <= 4):
                continue  # straight bar = letterbox edge, not a logo
            keep[lab == i] = 255
        if keep.any():
            keep = cv2.dilate(keep, np.ones((grow, grow), np.uint8))
            keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            mask[y:y + rh, x:x + rw] |= keep
            notes.append(f"ROI@{x},{y}: watermark candidate found")
    cover = (mask > 0).mean()
    if cover > 0.15:
        raise RuntimeError(f"Mask covers {cover:.0%} of the frame - that is not a watermark. Use --roi.")
    return mask, info, "\n".join(notes) or "nothing found"


def save_preview(path, mask, out_png, t=None):
    cap = cv2.VideoCapture(path)
    if t:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fr, cnts, -1, (0, 0, 255), 2)
    cv2.imwrite(out_png, fr)
