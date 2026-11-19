import cv2, numpy as np, sys
w, h, fps, secs = 1280, 720, 30, 20
vw = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
rng = np.random.default_rng(1)
base = cv2.GaussianBlur(rng.integers(0, 255, (h * 2, w * 2, 3), dtype=np.uint8), (0, 0), 6)
base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
for i in range(fps * secs):
    ox, oy = (i * 7) % w, (i * 4) % h
    fr = base[oy:oy + h, ox:ox + w].copy()
    cv2.putText(fr, "BRAND.COM", (w - 330, h - 40), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)
    vw.write(fr)
vw.release()
