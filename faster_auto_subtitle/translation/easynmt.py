import re
import time
import logging
from typing import List, Union
import torch
import numpy as np
import tqdm
import nltk

logger = logging.getLogger(__name__)

class EasyNMT:
    def __init__(self, model_name: str = None, translator=None, device=None):
        """
        Easy-to-use, state-of-the-art machine translation
        :param model_name:  Model name (see Readme for available models)
        :param translator: Translator object. Set to None, to automatically load the model via the model name.
        :param device: CPU / GPU device for PyTorch
        """
        self._model_name = model_name

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.device = device
        self.translator = translator


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
            start_time = time.time()
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
            #logger.info("Sentence splitting done after: {:.2f} sec".format(time.time() - start_time))
            #logger.info("Translate {} sentences".format(len(splitted_sentences)))

            translated_sentences = self.translate_sentences(splitted_sentences, target_lang=target_lang, source_lang=source_lang, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size, **kwargs)

            # Merge sentences back to documents
            start_time = time.time()
            translated_docs = []
            for doc_idx in range(len(documents)):
                start_idx = sent2doc[doc_idx - 1] if doc_idx > 0 else 0
                end_idx = sent2doc[doc_idx]
                translated_docs.append(self._reconstruct_document(documents[doc_idx], splitted_sentences[start_idx:end_idx], translated_sentences[start_idx:end_idx]))

            #logger.info("Document reconstruction done after: {:.2f} sec".format(time.time() - start_time))
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

    def translate_sentences(self, sentences: Union[str, List[str]], target_lang: str, source_lang: str = None,
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

    def sentence_splitting(self, text: str, lang: str = None):
        if lang == 'th':
            from thai_segmenter import sentence_segment # pylint: disable=C0415
            sentences = [str(sent) for sent in sentence_segment(text)]
        elif lang in ['ar', 'jp', 'ko', 'zh']:
            sentences = list(re.findall('[^!?。.]+[!?。.]*', text, flags=re.U))
        else:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            sentences = nltk.sent_tokenize(text)

        return sentences