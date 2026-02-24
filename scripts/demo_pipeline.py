"""Simulated pipeline demo for asciinema recording — no ML dependencies."""

from __future__ import annotations

import sys
import time

import ownscribe.progress as _prog
_prog._BRAILLE = "|/-\\"

from ownscribe.progress import PipelineProgress

OUT_DIR = "/Users/you/ownscribe/2026-02-24_1401"
OUT_DIR_RENAMED = "/Users/you/ownscribe/2026-02-24_1401_q1-planning-review"

SUMMARY_TEXT = """\
# Meeting Summary

## Summary
The team discussed Q1 deliverables and agreed on a March 15th deadline.
Design handoff is expected by March 1st, with the API freeze on March 8th.

## Key Points
- Backend migration to the new auth service is 80% complete
- Mobile team needs updated API docs before the freeze
- QA will start regression testing on March 10th

## Action Items
- [ ] Anna: Draft migration plan by Friday
- [ ] Bob: Finalize API docs before freeze
- [ ] Claire: Set up staging environment for QA

## Decisions
- Hard deadline for Q1 deliverables: March 15th
- Skip the optional analytics rework — revisit in Q2
"""


def _type_command(text: str) -> None:
    sys.stderr.write("$ ")
    sys.stderr.flush()
    for ch in text:
        sys.stderr.write(ch)
        sys.stderr.flush()
        time.sleep(0.04)
    sys.stderr.write("\n")
    sys.stderr.flush()
    time.sleep(0.3)


def _smooth_progress(progress: PipelineProgress, key: str, duration: float, steps: int = 50) -> None:
    dt = duration / steps
    for i in range(1, steps + 1):
        progress.update(key, i / steps)
        time.sleep(dt)


def _simulate_recording() -> None:
    sys.stderr.write("Starting recording... Press 'm' to mute/unmute mic, Ctrl+C to stop.\n\n")
    sys.stderr.flush()
    # Show a few timestamps to imply a recording, accelerated
    timestamps = [
        (0, 5), (0, 18), (0, 47), (1, 23), (3, 10), (5, 44), (8, 12),
    ]
    for mins, secs in timestamps:
        sys.stderr.write(f"\r  Recording: {mins:02d}:{secs:02d}\033[K")
        sys.stderr.flush()
        time.sleep(0.2)
    time.sleep(0.3)
    sys.stderr.write("\n\n")
    sys.stderr.write("Stopping recording...\n")
    sys.stderr.write(f"Audio saved to {OUT_DIR}/recording.wav\n\n")
    sys.stderr.flush()
    time.sleep(0.5)


def main() -> None:
    _type_command("ownscribe")
    _simulate_recording()

    with PipelineProgress(
        diarize=True,
        summarize=True,
        transcribe=True,
    ) as progress:
        # Transcribing — progress bar
        progress.begin("transcribing")
        _smooth_progress(progress, "transcribing", duration=4.0)
        progress.complete("transcribing")
        time.sleep(0.2)

        # Diarizing — sub-step progress bars
        progress.begin("diarizing")
        time.sleep(0.3)

        progress.begin("segmentation")
        _smooth_progress(progress, "segmentation", duration=1.5)
        progress.complete("segmentation")
        time.sleep(0.15)

        progress.begin("speaker_counting")
        time.sleep(0.15)
        progress.complete("speaker_counting")

        progress.begin("embeddings")
        _smooth_progress(progress, "embeddings", duration=1.5)
        progress.complete("embeddings")
        time.sleep(0.15)

        progress.begin("clustering")
        time.sleep(0.15)
        progress.complete("clustering")

        progress.complete("diarizing")
        time.sleep(0.2)

        # Summarizing — spinner only
        progress.begin("summarizing")
        time.sleep(2.5)
        progress.complete("summarizing")

    sys.stderr.write("\n")
    sys.stderr.write(f"Transcript saved to {OUT_DIR}/transcript.md\n")
    sys.stderr.write(f"Summary saved to {OUT_DIR}/summary.md\n\n")
    sys.stderr.write(SUMMARY_TEXT)
    sys.stderr.write(f"\nRecording deleted (keep_recording=false): {OUT_DIR_RENAMED}/recording.wav\n")
    sys.stderr.flush()
    time.sleep(2)


if __name__ == "__main__":
    main()
