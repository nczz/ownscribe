"""Simulated 'ownscribe ask' demo for asciinema recording — no ML dependencies."""

from __future__ import annotations

import sys
import time

import ownscribe.progress as _prog
_prog._BRAILLE = "|/-\\"

from ownscribe.progress import Spinner

ANSWER_TEXT = """\
The deadline for Q1 deliverables was set for March 15th.

> "The hard deadline for Q1 is March 15th." — SPEAKER_03, Q1 Planning Review [12:45]
> "We need everything wrapped by mid-March." — SPEAKER_02, Sprint Kickoff [05:30]
"""


def _type_command(text: str) -> None:
    sys.stderr.write("$ ")
    sys.stderr.flush()
    for ch in text:
        sys.stderr.write(ch)
        sys.stderr.flush()
        time.sleep(0.04)
    sys.stderr.write("\n\n")
    sys.stderr.flush()
    time.sleep(0.3)


def main() -> None:
    _type_command('ownscribe ask "What did we decide about the deadline?"')

    with Spinner("Searching 14 meetings"):
        time.sleep(2.0)

    sys.stderr.write(
        "Found 2 relevant meetings:\n"
        "  - 2026-02-24 14:30 — Q1 Planning Review\n"
        "  - 2026-02-20 10:15 — Sprint Kickoff\n\n"
    )
    sys.stderr.flush()
    time.sleep(0.5)

    with Spinner("Analyzing transcripts"):
        time.sleep(2.5)

    sys.stderr.write("\n")
    sys.stderr.write(ANSWER_TEXT)
    sys.stderr.flush()
    time.sleep(2)


if __name__ == "__main__":
    main()
