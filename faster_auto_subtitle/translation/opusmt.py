import time
import logging
import re
from copy import deepcopy
from typing import List, Optional, Union
import numpy as np
import tqdm
import nltk
import torch
from huggingface_hub import list_models
from transformers import MarianMTModel, MarianTokenizer
from faster_whisper.transcribe import Segment
from .languages import to_alpha2_languages, to_alpha3_language

logger = logging.getLogger(__name__)

NLP_ROOT = 'Helsinki-NLP'


class OpusMTWrapper:
    def __init__(self, device=None):
        """
        Easy-to-use, state-of-the-art machine translation
        :param model_name:  Model name (see Readme for available models)
        :param translator: Translator object. Set to None, to automatically load the model via the model name.
        :param device: CPU / GPU device for PyTorch
        """
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.translator = OpusMT()

    def translate_segments(self, segments: list[Segment], source_lang: str, target_lang: str) -> Optional[list[Segment]]:
        source_text = [segment.text for segment in segments]
        translation_available = self.translator.prepare_translation(source_lang, target_lang)
        if not translation_available:
            return None

        translated_text = self.translate(source_text, target_lang, source_lang, show_progress_bar=True)
        translated_segments = []
        for segment, translation in zip(segments, translated_text):
            new_segment = deepcopy(segment)
            new_segment.text = translation
            translated_segments.append(new_segment)
        return translated_segments

    def translate(self, documents: Union[str, List[str]], target_lang: str, source_lang: str,
                  show_progress_bar: bool = False, beam_size: int = 5, batch_size: int = 16,
                  perform_sentence_splitting: bool = True, paragraph_split: str = "\n", sentence_splitter=None,
                  **kwargs):
        """
        This method translates the given set of documents
        :param documents: If documents is a string, returns the translated document as string. If documents is a list of strings, translates all documents and returns a list.
        :param target_lang: Target language for the translation
        :param source_lang: Source language for all documents. If None, determines the source languages automatically.
        :param show_progress_bar: If true, plot a progress bar on the progress for the translation
        :param beam_size: Size for beam search
        :param batch_size: Number of sentences to translate at the same time
        :param perform_sentence_splitting: Longer documents are broken down sentences, which are translated individually
        :param paragraph_split: Split symbol for paragraphs. No sentences can go across the paragraph_split symbol.
        :param sentence_splitter: Method used to split sentences. If None, uses the default self.sentence_splitting method
        :param kwargs: Optional arguments for the translator model
        :return: Returns a string or a list of string with the translated documents
        """

        #Method_args will store all passed arguments to method
        method_args = locals()
        del method_args['self']
        del method_args['kwargs']
        method_args.update(kwargs)

        if source_lang == target_lang:
            return documents

        is_single_doc = False
        if isinstance(documents, str):
            documents = [documents]
            is_single_doc = True


        if perform_sentence_splitting:
            if sentence_splitter is None:
                sentence_splitter = self.sentence_splitting

            # Split document into sentences
            splitted_sentences = []
            sent2doc = []
            for doc in documents:
                paragraphs = doc.split(paragraph_split) if paragraph_split is not None else [doc]
                for para in paragraphs:
                    for sent in sentence_splitter(para.strip(), source_lang):
                        sent = sent.strip()
                        if len(sent) > 0:
                            splitted_sentences.append(sent)
                sent2doc.append(len(splitted_sentences))

            translated_sentences = self.translate_sentences(splitted_sentences, target_lang=target_lang, source_lang=source_lang, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size, **kwargs)

            # Merge sentences back to documents
            translated_docs = []
            for doc_idx, doc in enumerate(documents):
                start_idx = sent2doc[doc_idx - 1] if doc_idx > 0 else 0
                end_idx = sent2doc[doc_idx]
                translated_docs.append(self._reconstruct_document(doc, splitted_sentences[start_idx:end_idx], translated_sentences[start_idx:end_idx]))
        else:
            translated_docs = self.translate_sentences(documents, target_lang=target_lang, source_lang=source_lang, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size, **kwargs)

        if is_single_doc:
            translated_docs = translated_docs[0]

        return translated_docs

    @staticmethod
    def _reconstruct_document(doc, org_sent, translated_sent):
        """
        This method reconstructs the translated document and
        keeps white space in the beginning / at the end of sentences.
        """
        sent_idx = 0
        char_idx = 0
        translated_doc = ""
        while char_idx < len(doc):
            if sent_idx < len(org_sent) and doc[char_idx] == org_sent[sent_idx][0]:
                translated_doc += translated_sent[sent_idx]
                char_idx += len(org_sent[sent_idx])
                sent_idx += 1
            else:
                translated_doc += doc[char_idx]
                char_idx += 1
        return translated_doc

    def translate_sentences(self, sentences: Union[str, List[str]], target_lang: str, source_lang: str,
                            show_progress_bar: bool = False, beam_size: int = 5, batch_size: int = 32, **kwargs):
        """
        This method translates individual sentences.

        :param sentences: A single sentence or a list of sentences to be translated
        :param source_lang: Source language for all sentences. If none, determines automatically the source language
        :param target_lang: Target language for the translation
        :param show_progress_bar: Show a progress bar
        :param beam_size: Size for beam search
        :param batch_size: Mini batch size
        :return: List of translated sentences
        """

        if source_lang == target_lang:
            return sentences

        is_single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_sentence = True

        output = []

        #Sort by length to speed up processing
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        iterator = range(0, len(sentences_sorted), batch_size)
        if show_progress_bar:
            scale = min(batch_size, len(sentences))
            iterator = tqdm.tqdm(iterator, total=len(sentences)/scale, unit_scale=scale, smoothing=0)

        for start_idx in iterator:
            output.extend(self.translator.translate_sentences(sentences_sorted[start_idx:start_idx+batch_size], source_lang=source_lang, target_lang=target_lang, beam_size=beam_size, device=self.device, **kwargs))

        #Restore original sorting of sentences
        output = [output[idx] for idx in np.argsort(length_sorted_idx)]

        if is_single_sentence:
            output = output[0]

        return output

    def sentence_splitting(self, text: str, lang: str):
        if lang == 'th':
            from thai_segmenter import sentence_segment # pylint: disable=C0415
            sentences = [str(sent) for sent in sentence_segment(text)]
        elif lang in ['ar', 'jp', 'ko', 'zh']:
            sentences = list(re.findall('[^!?。.]+[!?。.]*', text, flags=re.U))
        else:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab')

            sentences = nltk.sent_tokenize(text)

        return sentences

class OpusMT:
    def __init__(self, max_loaded_models: int = 10):
        self.models: dict = {}
        self.max_loaded_models: int = max_loaded_models
        self.max_length: Optional[int] = None
        self.available_models: Optional[dict[str, DownloadableModel]] = None
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
        model_list = list_models(author=NLP_ROOT, model_name='opus-mt', tags=['marian'], sort='last_modified')

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
        if self.available_models is None:
            return []

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
        if self.available_models is None:
            return []

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
