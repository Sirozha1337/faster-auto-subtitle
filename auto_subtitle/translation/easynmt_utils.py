from easynmt import EasyNMT
from faster_whisper.transcribe import Segment
from .opusmt_utils import OpusMT


class EasyNMTWrapper:
    def __init__(self, device: str):
        self.translator = OpusMT()
        self.model = EasyNMT('opus-mt',
                             translator=self.translator,
                             device=device if device != 'auto' else None)

    def translate(self, segments: list[Segment], source_lang: str, target_lang: str) -> list[Segment]:
        source_text = [segment.text for segment in segments]
        self.translator.load_available_models()

        translated_text = self.model.translate(source_text, target_lang,
                                               source_lang, show_progress_bar=True)
        translated_segments = list()
        for segment, translation in zip(segments, translated_text):
            translated_segments.append(segment._replace(text=translation))

        return translated_segments
