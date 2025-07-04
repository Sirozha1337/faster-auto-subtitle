from typing import Optional
from copy import deepcopy
from faster_whisper.transcribe import Segment
from .easynmt import EasyNMT
from .opusmt_utils import OpusMT

class EasyNMTWrapper:
    def __init__(self, device: str):
        self.translator = OpusMT()
        self.model = EasyNMT('opus-mt',
                             translator=self.translator,
                             device=device if device != 'auto' else None)

    def translate(self, segments: list[Segment], source_lang: str,
                  target_lang: str) -> Optional[list[Segment]]:
        source_text = [segment.text for segment in segments]
        translation_available = self.translator.prepare_translation(source_lang, target_lang)
        if not translation_available:
            return None

        translated_text = self.model.translate(source_text, target_lang, source_lang, show_progress_bar=True)
        translated_segments = []
        for segment, translation in zip(segments, translated_text):
            new_segment = deepcopy(segment)
            new_segment.text = translation
            translated_segments.append(new_segment)

        return translated_segments
