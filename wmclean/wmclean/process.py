"""Chunked, parallel, encode-once pipeline. All heavy lifting is inside ffmpeg."""
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .detect import probe


def _encoder_args(nvenc, preset="veryfast"):
    if nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p2", "-cq", "21"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", "21"]


def run_chunk(src, mask_png, out, start, dur, nvenc=False, threads=0, preset="veryfast"):
    """Process one time-slice. Input-seek + re-encode = frame-accurate cuts."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
           "-vf", f"removelogo={mask_png}",
           *_encoder_args(nvenc, preset), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k",
           "-threads", str(threads), "-avoid_negative_ts", "make_zero", out]
    subprocess.check_call(cmd)


def plan_chunks(duration, n):
    step = duration / n
    return [(i * step, step if i < n - 1 else duration - i * step) for i in range(n)]


def merge(parts, out):
    d = tempfile.mkdtemp()
    lst = os.path.join(d, "list.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                           "-safe", "0", "-i", lst, "-c", "copy",
                           "-movflags", "+faststart", out])
    shutil.rmtree(d, ignore_errors=True)


def run_local(src, mask_png, out, workers=None, chunks=None, nvenc=False):
    info = probe(src)
    workers = workers or os.cpu_count() or 1
    chunks = chunks or workers
    tmp = tempfile.mkdtemp(prefix="wmclean_")
    jobs = plan_chunks(info["duration"], chunks)
    parts = [os.path.join(tmp, f"part_{i:04d}.mp4") for i in range(len(jobs))]
    per = max(1, (os.cpu_count() or 1) // workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_chunk, src, mask_png, p, s, d, nvenc, per)
                for p, (s, d) in zip(parts, jobs)]
        for f in futs:
            f.result()
    merge(parts, out)
    shutil.rmtree(tmp, ignore_errors=True)
