import tempfile
import os
import shutil
from typing import TextIO, cast
from faster_auto_subtitle.models.Subtitles import Subtitles
from faster_auto_subtitle.utils.files import write_srt


class SubtitlesTempFile:
    def __init__(self, subtitles: Subtitles):
        self.subtitles = subtitles
        self.tmp_file = None
        self.tmp_file_path = None

    def __enter__(self):
        if self.subtitles is None:
            return self

        self.tmp_file = tempfile.NamedTemporaryFile('w', encoding="utf-8", dir='.', delete=False)
        self.tmp_file_path = os.path.relpath(self.tmp_file.name, '.')

        if self.subtitles.output_path is not None and os.path.isfile(self.subtitles.output_path):
            shutil.copyfile(self.subtitles.output_path, self.tmp_file_path)
        else:
            write_srt(self.subtitles.segments, cast(TextIO, self.tmp_file))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if self.subtitles is None:
            return

        self.tmp_file.close()
        if os.path.isfile(self.tmp_file_path):
            os.remove(self.tmp_file_path)
