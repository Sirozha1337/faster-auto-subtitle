from typing import Optional
from faster_whisper.transcribe import Segment


class Subtitles:
    segments: list
    language: str
    output_path: Optional[str] = None

    def __init__(self, segments: list[Segment], language: str):
        self.language = language
        self.segments = segments
