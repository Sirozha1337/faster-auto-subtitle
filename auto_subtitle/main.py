import os
import warnings
from .utils.files import filename, write_srt
from .utils.ffmpeg import get_audio, overlay_subtitles
from .utils.whisper import WhisperAI
from .translation.easynmt_utils import EasyNMTWrapper


def process(args: dict):
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    sample_interval: list = args.pop("sample_interval")
    target_language: str = args.pop("target_language")

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
        language = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language

    if target_language != 'en':
        warnings.warn(
            f"{target_language} is not English, Opus-MT will be used to perform translation.")
        args['task'] = 'transcribe'

    audios = get_audio(args.pop("video"), args.pop(
        'audio_channel'), sample_interval)

    model_args = {
        "model_size_or_path": model_name,
        "device": args.pop("device"),
        "compute_type": args.pop("compute_type")
    }

    subtitles = get_subtitles(audios, model_args, args)
    print('Subtitles generated.')

    if target_language != 'en':
        print('Translating subtitles... This might take a while.')
        subtitles = translate_subtitles(
            subtitles, language, target_language, model_args)

    if output_srt or srt_only:
        print('Saving subtitle files...')
        save_subtitles(subtitles, output_dir)

    if srt_only:
        return

    overlay_subtitles(subtitles, output_dir, sample_interval)


def translate_subtitles(subtitles: dict, source_lang: str, target_lang: str, model_args: dict):
    model = EasyNMTWrapper(device=model_args['device'])

    translated_subtitles = {}
    for key, subtitle in subtitles.items():
        src_lang = source_lang
        if src_lang == '' or src_lang is None:
            src_lang = subtitle['language']

        translated_segments = model.translate(
            subtitle['segments'], src_lang, target_lang)

        translated_subtitle = subtitle.copy()
        translated_subtitle['segments'] = translated_segments
        translated_subtitles[key] = translated_subtitle

    return translated_subtitles


def save_subtitles(subtitles: dict, output_dir: str):
    for path, subtitle in subtitles.items():
        subtitle["output_path"] = os.path.join(
            output_dir, f"{filename(path)}.srt")

        print(f'Saving to path {subtitle["output_path"]}')
        with open(subtitle['output_path'], "w", encoding="utf-8") as srt:
            write_srt(subtitle['segments'], file=srt)


def get_subtitles(audio_paths: dict, model_args: dict, transcribe_args: dict):
    model = WhisperAI(model_args, transcribe_args)

    subtitles = {}

    for path, audio_path in audio_paths.items():
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        segments, info = model.transcribe(audio_path)

        subtitles[path] = {'segments': list(
            segments), 'language': info.language}

    return subtitles
