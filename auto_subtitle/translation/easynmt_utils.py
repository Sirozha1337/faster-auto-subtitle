from easynmt import EasyNMT
from faster_whisper.transcribe import Segment
from .opusmt_utils import OpusMT


class EasyNMTWrapper:
    def __init__(self, device):
        self.translator = OpusMT()
        self.model = EasyNMT('opus-mt',
                             translator=self.translator,
                             device=device if device != 'auto' else None)

    def translate(self, segments: list[Segment], source_lang: str, target_lang: str):
        source_text = [segment.text for segment in segments]
        self.translator.load_available_models()

        translated_text = self.model.translate(source_text, target_lang,
                                               source_lang, show_progress_bar=True)
        translated_segments = [None] * len(segments)
        for index, segment in enumerate(segments):
            translated_segments[index] = segment._replace(
                text=translated_text[index])

        return translated_segments
