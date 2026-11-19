import argparse
import sys
import time

import cv2

from . import detect, process


def _roi(s):
    return tuple(int(v) for v in s.split(",")) if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wmclean")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="find watermark mask + preview image")
    d.add_argument("video"); d.add_argument("--mask", default="mask.png")
    d.add_argument("--preview", default="preview.png"); d.add_argument("--roi", help="x,y,w,h")

    c = sub.add_parser("chunk", help="process one time slice (used by CI fan-out)")
    c.add_argument("video"); c.add_argument("--mask", required=True)
    c.add_argument("--out", required=True); c.add_argument("--start", type=float, required=True)
    c.add_argument("--dur", type=float, required=True); c.add_argument("--nvenc", action="store_true")
    c.add_argument("--preset", default="ultrafast")

    pl = sub.add_parser("plan", help="print JSON chunk matrix for CI")
    pl.add_argument("video"); pl.add_argument("--chunks", type=int, default=20)

    m = sub.add_parser("merge"); m.add_argument("--out", required=True); m.add_argument("parts", nargs="+")

    r = sub.add_parser("run", help="detect + process everything locally in parallel")
    r.add_argument("video"); r.add_argument("--out", required=True)
    r.add_argument("--roi"); r.add_argument("--workers", type=int); r.add_argument("--nvenc", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "detect":
        mask, info, rep = detect.detect_mask(a.video, _roi(a.roi))
        print(rep)
        if not mask.any():
            sys.exit("No watermark detected. Pass --roi x,y,w,h to force a region.")
        cv2.imwrite(a.mask, mask); detect.save_preview(a.video, mask, a.preview)
        print(f"mask -> {a.mask}, check {a.preview} before burning compute")
    elif a.cmd == "chunk":
        process.run_chunk(a.video, a.mask, a.out, a.start, a.dur, a.nvenc, 0, a.preset)
    elif a.cmd == "plan":
        import json
        inf = detect.probe(a.video)
        print(json.dumps([{"i": i, "start": round(s, 3), "dur": round(d, 3)}
                          for i, (s, d) in enumerate(process.plan_chunks(inf["duration"], a.chunks))]))
    elif a.cmd == "merge":
        process.merge(a.parts, a.out)
    elif a.cmd == "run":
        t = time.time()
        mask, info, rep = detect.detect_mask(a.video, _roi(a.roi))
        print(rep)
        if not mask.any():
            sys.exit("No watermark detected. Pass --roi x,y,w,h.")
        cv2.imwrite("mask.png", mask); detect.save_preview(a.video, mask, "preview.png")
        process.run_local(a.video, "mask.png", a.out, a.workers, nvenc=a.nvenc)
        print(f"done in {time.time() - t:.1f}s for {info['duration']:.0f}s of video")


if __name__ == "__main__":
    main()
