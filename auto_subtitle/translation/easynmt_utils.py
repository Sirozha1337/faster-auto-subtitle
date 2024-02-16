from typing import Optional
from easynmt import EasyNMT
from faster_whisper.transcribe import Segment
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

        translated_text = self.model.translate(source_text, target_lang, source_lang,
                                               document_language_detection=False,
                                               show_progress_bar=True)
        translated_segments = []
        for segment, translation in zip(segments, translated_text):
            translated_segments.append(segment._replace(text=translation))

        return translated_segments
