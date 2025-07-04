import os
from typing import TextIO, Iterator
from faster_whisper.transcribe import Segment
from .convert import format_timestamp


def write_srt(transcript: Iterator[Segment], file: TextIO) -> None:
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment.start, always_include_hours=True)} --> "
            f"{format_timestamp(segment.end, always_include_hours=True)}\n"
            f"{segment.text.strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
