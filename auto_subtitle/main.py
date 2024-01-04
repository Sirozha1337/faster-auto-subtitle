import os
import warnings
import tempfile
from .utils.files import filename, write_srt
from .utils.ffmpeg import get_audio, overlay_subtitles
from .utils.whisper import WhisperAI

def process(args: dict):
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    sample_interval: str = args.pop("sample_interval")
    
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language

    audios = get_audio(args.pop("video"), args.pop('audio_channel'), sample_interval)
    subtitles = get_subtitles(
        audios, output_srt or srt_only, output_dir, model_name, args
    )

    if srt_only:
        return

    overlay_subtitles(subtitles, output_dir, sample_interval)

def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, model_name: str, model_args: dict):
    model = WhisperAI(model_name, model_args)

    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        segments = model.transcribe(audio_path)

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(segments, file=srt)

        subtitles_path[path] = srt_path

    return subtitles_path