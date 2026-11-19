# wmclean — fast static-watermark remover

Detects a static watermark/logo automatically, then removes it with an encode-once,
chunked, parallel ffmpeg pipeline. Built for speed on long videos (3 h), not for miracles.

> Only use this on video you own or have the rights to edit (your own logos, your own
> exports, licensed footage). Stripping someone else's attribution is on you.

## Honest limits (read this)

| Case | Result |
|---|---|
| Static opaque logo/text in a corner | Works, fast. Leaves a soft smudge on busy backgrounds. |
| Semi-transparent static logo | Detected only if its edges are strong. Often partial. |
| Moving / animated / full-screen tiled watermark | **Not supported.** Needs per-frame deep inpainting = hours, not minutes. |
| Static-camera footage (studio, CCTV) | Background looks "static" too; use `--roi`. |
| Quality like ProPainter | Not on a 5-minute budget. Physics says no. |

## Why 3 h in < 5 min needs fan-out

3 h x 30 fps = 324,000 frames -> ~1,080 fps sustained. One CPU box does ~40-100 fps.
So: split into 20 time chunks, process them on 20 machines at once, concat with `-c copy`.

## Run from GitHub (nothing saved on your machine)

1. Push this repo (public repo = free, 4-core runners; private = 2-core, slower).
2. Put the video somewhere with a **direct, seekable HTTPS URL** (Cloudflare R2 / S3
   presigned GET). Git can't hold it (100 MB limit), Releases cap at 2 GB, and Google
   Drive share links aren't seekable.
3. Actions -> `remove-watermark` -> Run workflow -> paste `video_url`
   (and a presigned PUT `upload_url` so the result goes straight to your bucket).
4. Check `preview.png` in the `mask` artifact. Red outline = what gets removed.

Each chunk job reads only its slice over HTTP range requests; nothing is downloaded whole.

## Local use

```bash
pip install -r requirements.txt     # needs ffmpeg on PATH
python -m wmclean detect in.mp4     # writes mask.png + preview.png
python -m wmclean run in.mp4 --out clean.mp4 --workers 8
python -m wmclean run in.mp4 --out clean.mp4 --nvenc   # NVIDIA GPU encode
```

## Hitting the 5-minute target

Time ~= frames / (jobs x fps_per_job). Measured here: ~35 fps on a *single* weak core
at 720p. Benchmark one chunk first, then pick the chunk count:

- 1080p30, 4-core runner, `ultrafast`: ~80-120 fps/job (verify!) -> 20 jobs ~ 3-4 min.
- 4K or 60 fps: 4x-8x the work. Free Actions will NOT make 5 min. Use GPU workers
  (RunPod/Modal/Lambda, `--nvenc`) with 40+ chunks.
- Runner boot + apt + upload add ~1 min. Budget for it.

## Roadmap if you want better quality

Keep the same chunk/fan-out skeleton, swap `removelogo` for a crop-only LaMa pass on GPU
(only the mask bounding box, batched). Expect ~5-10x slower; needs GPU workers.
