import os
import warnings
import logging
from typing import Optional
from .models.Subtitles import Subtitles, SegmentsIterable
from .utils.files import filename, write_srt
from .utils.ffmpeg import get_audio, add_subtitles, preprocess_audio
from .utils.whisper import WhisperAI
from .translation.easynmt_utils import EasyNMTWrapper

logger = logging.getLogger(__name__)


def process(args: dict):
    model_name: str = args.pop("model")
    language: str = args.pop("language")
    sample_interval: list = args.pop("sample_interval")
    target_language: str = args.pop("target_language")

    logging.basicConfig(encoding='utf-8', level=logging.INFO)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
        language = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language

    if target_language != 'en':
        logger.info("%s is not English, Opus-MT will be used to perform translation.",
                    target_language)
        args['task'] = 'transcribe'

    output_args = {
        "output_dir": args.pop("output_dir"),
        "output_type": args.pop("output_type"),
        "subtitle_type": args.pop("subtitle_type")
    }

    videos = args.pop('video')
    audio_channel = args.pop('audio_channel')
    model_args = {
        "model_size_or_path": model_name,
        "device": args.pop("device"),
        "compute_type": args.pop("compute_type")
    }
    transcribe_model = WhisperAI(model_args, args)
    translate_model = EasyNMTWrapper(
        device=model_args['device']) if target_language != 'en' else None

    os.makedirs(output_args["output_dir"], exist_ok=True)
    for video in videos:
        if video.endswith('.wav'):
            audio = preprocess_audio(video, audio_channel, sample_interval)
        else:
            audio = get_audio(video, audio_channel, sample_interval)

        transcribed, translated = perform_task(video, audio, language, target_language,
                                               transcribe_model, translate_model)

        save_result(video, transcribed, translated, sample_interval, output_args)


def save_result(video: str, transcribed: Subtitles, translated: Subtitles, sample_interval: list,
                output_args: dict[str, str]) -> None:
    if output_args["output_type"] == 'all' or output_args["output_type"] == 'srt':
        logger.info('Saving subtitle files...')
        save_subtitles(video, transcribed, output_args["output_dir"], translated is not None)

        if translated is not None:
            save_subtitles(video, translated, output_args["output_dir"], translated is not None)

    if output_args["output_type"] == 'srt':
        return

    add_subtitles(video, transcribed, translated, sample_interval, output_args)


def perform_task(video: str, audio: str, language: str, target_language: str,
                 transcribe_model: WhisperAI,
                 translate_model: EasyNMTWrapper) -> tuple[Subtitles, Optional[Subtitles]]:
    transcribed = get_subtitles(video, audio, transcribe_model)
    translated = None

    logger.info('Subtitles generated.')
    if target_language != 'en':
        logger.info('Translating subtitles... This might take a while.')
        translated = translate_subtitles(
            transcribed, language, target_language, translate_model)

    return transcribed, translated


def translate_subtitles(subtitles: Subtitles, source_lang: str, target_lang: str,
                        model: EasyNMTWrapper) -> Subtitles:
    src_lang = source_lang
    if src_lang == '' or src_lang is None:
        src_lang = subtitles.language

    translated_segments = model.translate(
        list(subtitles.segments), src_lang, target_lang)

    return Subtitles(SegmentsIterable(translated_segments), target_lang)


def save_subtitles(path: str, subtitles: Subtitles, output_dir: str,
                   use_language_in_output: bool) -> None:
    if use_language_in_output:
        subtitles.output_path = os.path.join(
            output_dir, f"{filename(path)}.{subtitles.language}.srt")
    else:
        subtitles.output_path = os.path.join(
            output_dir, f"{filename(path)}.srt")

    logger.info('Saving to path %s', subtitles.output_path)
    with open(subtitles.output_path, "w", encoding="utf-8") as srt:
        write_srt(subtitles.segments, file=srt)


def get_subtitles(source_path: str, audio_path: str, model: WhisperAI) -> Subtitles:
    logger.info("Generating subtitles for %s... This might take a while.",
                filename(source_path))

    segments, info = model.transcribe(audio_path)

    return Subtitles(segments=SegmentsIterable(segments), language=info.language)
