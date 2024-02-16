from typing import Optional, Iterable, Iterator
from faster_whisper.transcribe import Segment


class SegmentsIterable(Iterable):
    segments: list = None
    index: int = None
    length: int = 0
    __segments_iterable: Iterator[Segment]

    def __init__(self, segments: Iterable[Segment]):
        self.__segments_iterable = segments.__iter__()
        self.segments = []

    def __iter__(self):
        if self.index is not None:
            self.index = 0
        return self

    def __next_list(self):
        if self.index < self.length:
            item = self.segments[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration

    def __next_iter(self):
        try:
            item = next(self.__segments_iterable)
            self.segments.append(item)
            self.length += 1
            return item
        except StopIteration:
            self.index = 0
            raise StopIteration

    def __next__(self):
        if self.index is not None:
            return self.__next_list()

        return self.__next_iter()


class Subtitles:
    segments: SegmentsIterable[Segment]
    language: str
    output_path: Optional[str] = None

    def __init__(self, segments: SegmentsIterable[Segment], language: str):
        self.language = language
        self.segments = segments
