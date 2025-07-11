from typing import Optional
from copy import deepcopy
from faster_whisper.transcribe import Segment
try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator, DeeplTranslator, QcriTranslator, LingueeTranslator, PonsTranslator, YandexTranslator, MicrosoftTranslator, PapagoTranslator, LibreTranslator, BaiduTranslator  # type: ignore
except ImportError:
    # These will be None if deep-translator is not installed, but this allows static analysis to pass
    GoogleTranslator = MyMemoryTranslator = DeeplTranslator = QcriTranslator = LingueeTranslator = PonsTranslator = YandexTranslator = MicrosoftTranslator = PapagoTranslator = LibreTranslator = BaiduTranslator = None

TRANSLATOR_MAP = {
    'google': GoogleTranslator,
    'mymemory': MyMemoryTranslator,
    'deepl': DeeplTranslator,
    'qcri': QcriTranslator,
    'linguee': LingueeTranslator,
    'pons': PonsTranslator,
    'yandex': YandexTranslator,
    'microsoft': MicrosoftTranslator,
    'papago': PapagoTranslator,
    'libre': LibreTranslator,
    'baidu': BaiduTranslator,
}

class DeepTranslatorWrapper:
    def __init__(self, mode: str = 'google', **kwargs):
        if mode not in TRANSLATOR_MAP or TRANSLATOR_MAP[mode] is None:
            raise ValueError(f"Unknown or unavailable deep-translator mode: {mode}")
        self.mode = mode
        self.translator_class = TRANSLATOR_MAP[mode]
        self.kwargs = kwargs
        self.translator = None

    def translate_segments(self, segments: list[Segment], source_lang: str, target_lang: str) -> Optional[list[Segment]]:
        if self.translator_class is None or not callable(self.translator_class):
            raise ImportError("deep-translator is not installed or the selected mode is unavailable.")
            
        source_text = [segment.text for segment in segments]
        # Instantiate translator
        translator = self.translator_class(source=source_lang, target=target_lang, **self.kwargs)

        # Use batch translation if available
        if hasattr(translator, 'translate_batch'):
            translated_text = translator.translate_batch(source_text)
        else:
            translated_text = [translator.translate(text) for text in source_text]

        translated_segments = []
        for segment, translation in zip(segments, translated_text):
            new_segment = deepcopy(segment)
            new_segment.text = translation
            translated_segments.append(new_segment)
        return translated_segments
