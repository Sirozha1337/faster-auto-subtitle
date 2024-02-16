import time
import logging
from typing import List, Optional
import torch
from huggingface_hub import list_models, ModelFilter
from transformers import MarianMTModel, MarianTokenizer
from .languages import to_alpha2_languages, to_alpha3_language

logger = logging.getLogger(__name__)

NLP_ROOT = 'Helsinki-NLP'


class OpusMT:
    def __init__(self, max_loaded_models: int = 10):
        self.models: dict = {}
        self.max_loaded_models: int = max_loaded_models
        self.max_length: Optional[int] = None
        self.available_models: Optional[dict] = None
        self.prepared_translations: dict = {}

    def load_model(self, model_name: str) -> tuple:
        if model_name in self.models:
            self.models[model_name]['last_loaded'] = time.time()
            return self.models[model_name]['tokenizer'], self.models[model_name]['model']

        logger.info("Load model: %s", model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()

        if len(self.models) >= self.max_loaded_models:
            oldest_time = time.time()
            oldest_model = None
            for loaded_model_name, loaded_model in self.models.items():
                if loaded_model['last_loaded'] <= oldest_time:
                    oldest_model = loaded_model_name
                    oldest_time = loaded_model['last_loaded']
            del self.models[oldest_model]

        self.models[model_name] = {
            'tokenizer': tokenizer, 'model': model, 'last_loaded': time.time()}
        return tokenizer, model

    def load_available_models(self) -> None:
        if self.available_models is not None:
            return

        logger.info('Loading a list of available language models from OPUS-MT')
        model_list = list_models(
            filter=ModelFilter(
                author=NLP_ROOT,
                model_name='opus-mt'
            )
        )

        suffix = [x.modelId.split("/")[1] for x in model_list
                  if x.modelId.startswith(f'{NLP_ROOT}/opus-mt') and 'tc' not in x.modelId]

        models = [DownloadableModel(f"{NLP_ROOT}/{s}")
                  for s in suffix if s == s.lower()]

        self.available_models = {}
        for model in models:
            for src in model.source_languages:
                for tgt in model.target_languages:
                    key = self.make_translation_key(src, tgt)
                    if key not in self.available_models:
                        self.available_models[key] = model
                    elif self.available_models[key].language_count > model.language_count:
                        self.available_models[key] = model

    @staticmethod
    def make_translation_key(source_lang: str, target_lang: str) -> str:
        return f'{source_lang}-{target_lang}'

    def prepare_translation(self, source_lang: str, target_lang: str) -> bool:
        self.load_available_models()

        translation_key = self.make_translation_key(source_lang, target_lang)
        if translation_key in self.prepared_translations:
            return self.prepared_translations[translation_key]

        translations = self.determine_required_translations(source_lang, target_lang)
        if len(translations) == 0:
            return False

        self.prepared_translations[translation_key] = translations
        return True

    def determine_required_translations(self, source_lang: str, target_lang: str) -> List[tuple]:
        direct_key = self.make_translation_key(source_lang, target_lang)
        if direct_key in self.available_models:
            logger.info(
                'Found direct translation from %s to %s.', source_lang, target_lang)
            return [(source_lang, target_lang, direct_key)]

        logger.info(
            'No direct translation from %s to %s. Trying to translate through en.', source_lang,
            target_lang)

        to_en_key = self.make_translation_key(source_lang, 'en')
        if to_en_key not in self.available_models:
            logger.warning('No translation from %s to en.', source_lang)
            return []

        from_en_key = self.make_translation_key('en', target_lang)
        if from_en_key not in self.available_models:
            logger.warning('No translation from en to %s.', target_lang)
            return []

        return [(source_lang, 'en', to_en_key), ('en', target_lang, from_en_key)]

    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str,
                            device: str, beam_size: int = 5, **kwargs) -> List[str]:

        translations = []
        translation_key = self.make_translation_key(source_lang, target_lang)
        if translation_key in self.prepared_translations:
            translations = self.prepared_translations[translation_key]
        else:
            logger.warning('prepare_translation method should be called prior to '
                           'translate_sentences')

        intermediate = sentences
        for _, intermediate_target_language, key in translations:
            model_data = self.available_models[key]
            model_name = model_data.name
            tokenizer, model = self.load_model(model_name)
            model.to(device)

            # MultiLanguage model requires prepending each line with target language
            if model_data.multilanguage:
                alpha3 = to_alpha3_language(intermediate_target_language)
                prefix = next(
                    x for x in tokenizer.supported_language_codes if alpha3 in x)
                intermediate = [f'{prefix} {x}' for x in intermediate]

            inputs = tokenizer(intermediate, truncation=True, padding=True,
                               max_length=self.max_length, return_tensors="pt")

            for token in inputs:
                inputs[token] = inputs[token].to(device)

            with torch.no_grad():
                translated = model.generate(
                    **inputs, num_beams=beam_size, **kwargs)
                intermediate = [tokenizer.decode(
                    t, skip_special_tokens=True) for t in translated]

        return intermediate


class DownloadableModel:
    def __init__(self, name: str):
        self.name = name
        source_languages, target_languages = self.parse_languages(name)
        self.source_languages = source_languages
        self.target_languages = target_languages
        self.multilanguage = len(self.target_languages) > 1
        self.language_count = len(
            self.source_languages) + len(self.target_languages)

    @staticmethod
    def parse_languages(name: str) -> tuple[set, set]:
        parts = name.split('-')
        if len(parts) > 5:
            return set(), set()

        src, tgt = parts[3], parts[4]
        return to_alpha2_languages(src.split('_')), to_alpha2_languages(tgt.split('_'))
