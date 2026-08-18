"""
Clip Generator Backend
-----------------------
Takes a YouTube link, downloads the video, transcribes it (free, local model),
picks a handful of good moments, cuts them into vertical 9:16 clips with
burned-in captions, and serves them back for download.

Deploy this on Render.com (see README.md). No paid API key required.
"""

import os
import re
import uuid
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Clip Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(JOBS_DIR)), name="files")

print("Loading transcription model... (first run may take a minute)")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded.")


class ClipRequest(BaseModel):
    youtube_url: str
    num_clips: int = 5
    clip_length_seconds: int = 45


HOOK_WORDS = [
    "secret", "never", "worst", "best", "shocking", "crazy", "insane",
    "nobody", "everyone", "mistake", "truth", "why", "how", "stop",
    "warning", "important", "changed my life", "biggest", "huge",
]


def run(cmd: list[str]):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def download_video(url: str, out_path: Path) -> Path:
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        url,
    ]
    run(cmd)
    if not out_path.exists():
        raise RuntimeError("Download failed - file not found after yt-dlp run.")
    return out_path


def transcribe(video_path: Path):
    segments, _ = model.transcribe(str(video_path), beam_size=5, vad_filter=True)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]


def score_window(segs: list[dict]) -> float:
    text = " ".join(s["text"] for s in segs).lower()
    score = len(text.split()) * 0.1
    score += sum(3 for w in HOOK_WORDS if w in text)
    score += text.count("?") * 2
    score += text.count("!") * 1
    return score


def pick_clips(segments: list[dict], num_clips: int, clip_len: int):
    if not segments:
        return []

    total_end = segments[-1]["end"]
    candidates = []
    step = max(10, clip_len // 2)
    t = 0
    while t + clip_len <= total_end:
        window_segs = [s for s in segments if s["start"] >= t and s["start"] < t + clip_len]
        if window_segs:
            candidates.append({
                "start": window_segs[0]["start"],
                "end": window_segs[-1]["end"],
                "score": score_window(window_segs),
                "segs": window_segs,
            })
        t += step

    candidates.sort(key=lambda c: c["score"], reverse=True)

    chosen = []
    for c in candidates:
        overlap = any(not (c["end"] < ch["start"] or c["start"] > ch["end"]) for ch in chosen)
        if not overlap:
            chosen.append(c)
        if len(chosen) >= num_clips:
            break

    chosen.sort(key=lambda c: c["start"])
    return chosen


def write_srt(segs: list[dict], clip_start: float, out_path: Path):
    def fmt(t):
        t = max(0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segs, start=1):
        start = seg["start"] - clip_start
        end = seg["end"] - clip_start
        lines.append(str(i))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(seg["text"])
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def cut_and_caption(video_path: Path, clip: dict, out_path: Path, srt_path: Path):
    duration = clip["end"] - clip["start"]
    vf = (
        "scale=-2:1920,crop=1080:1920,"
        f"subtitles={srt_path.name}:force_style="
        "'FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=3,Outline=2,Alignment=2,MarginV=80'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip["start"]),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(srt_path.parent))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


@app.post("/process")
def process_video(req: ClipRequest):
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if not re.match(r"^https?://", req.youtube_url):
        raise HTTPException(400, "Please provide a valid video URL.")

    try:
        source_path = job_dir / "source.mp4"
        download_video(req.youtube_url, source_path)

        segments = transcribe(source_path)
        if not segments:
            raise HTTPException(422, "Could not transcribe any speech from this video.")

        clips = pick_clips(segments, req.num_clips, req.clip_length_seconds)
        if not clips:
            raise HTTPException(422, "Video too short to generate clips.")

        results = []
        for idx, clip in enumerate(clips, start=1):
            clip_segs = clip["segs"]
            srt_path = job_dir / f"clip_{idx}.srt"
            write_srt(clip_segs, clip["start"], srt_path)

            out_name = f"clip_{idx}.mp4"
            out_path = job_dir / out_name
            cut_and_caption(source_path, clip, out_path, srt_path)

            results.append({
                "clip": idx,
                "start": round(clip["start"], 1),
                "end": round(clip["end"], 1),
                "preview_text": " ".join(s["text"] for s in clip_segs)[:120],
                "download_url": f"/files/{job_id}/{out_name}",
            })

        source_path.unlink(missing_ok=True)

        return {"job_id": job_id, "clips": results}

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Something went wrong: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
