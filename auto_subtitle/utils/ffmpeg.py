import os
import tempfile
import ffmpeg
import logging
from .tempfile import SubtitlesTempFile
from .files import filename
from ..models.Subtitles import Subtitles
from typing import Optional

logger = logging.getLogger(__name__)


def get_audio(path: str, audio_channel_index: int, sample_interval: list):
    temp_dir = tempfile.gettempdir()

    logger.info(f"Extracting audio from {filename(path)}...")
    output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

    ffmpeg_input_args = {}
    if sample_interval is not None:
        ffmpeg_input_args['ss'] = str(sample_interval[0])

    ffmpeg_output_args = {
        'acodec': "pcm_s16le",
        'ac': "1",
        'ar': "16k",
        'map': "0:a:" + str(audio_channel_index)
    }
    if sample_interval is not None:
        ffmpeg_output_args['t'] = str(
            sample_interval[1] - sample_interval[0])

    ffmpeg.input(path, **ffmpeg_input_args).output(
        output_path,
        **ffmpeg_output_args
    ).run(quiet=True, overwrite_output=True)

    return output_path


def add_subtitles(path: str, transcribed: Subtitles, translated: Optional[Subtitles],
                  output_dir: str, sample_interval: list, subtitle_type: str):
    out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

    logger.info(f"Adding subtitles to {filename(path)}...")

    ffmpeg_input_args = {}
    if sample_interval is not None:
        ffmpeg_input_args['ss'] = str(sample_interval[0])

    ffmpeg_output_args = {}
    if sample_interval is not None:
        ffmpeg_output_args['t'] = str(
            sample_interval[1] - sample_interval[0])

    # HACK: On Windows it's impossible to use absolute subtitle file path with ffmpeg
    # so we use temp copy instead
    # see: https://github.com/kkroening/ffmpeg-python/issues/745
    with SubtitlesTempFile(transcribed) as transcribed_tmp, SubtitlesTempFile(translated) as translated_tmp:

        if subtitle_type == 'hard':
            hard_subtitles(path, out_path, transcribed_tmp, translated_tmp, ffmpeg_input_args, ffmpeg_output_args)
        elif subtitle_type == 'soft':
            soft_subtitles(path, out_path, transcribed_tmp, translated_tmp, ffmpeg_input_args, ffmpeg_output_args)

    logger.info(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def hard_subtitles(input_path: str, output_path: str,
                   transcribed: SubtitlesTempFile, translated: SubtitlesTempFile,
                   input_args: dict, output_args: dict):
    video = ffmpeg.input(input_path, **input_args)
    audio = video.audio

    intermediate = video.filter(
        'subtitles', transcribed.tmp_file_path,
        force_style="OutlineColour=&H40000000,BorderStyle=3")

    if translated.tmp_file is not None:
        intermediate.filter(
            'subtitles', translated.tmp_file_path,
            force_style="OutlineColour=&H40000000,BorderStyle=3,Alignment=6")
    ffmpeg.concat(
        intermediate, audio, v=1, a=1
    ).output(output_path, **output_args) \
        .run(quiet=True, overwrite_output=True)


def soft_subtitles(input_path: str, output_path: str,
                   transcribed: SubtitlesTempFile, translated: SubtitlesTempFile,
                   input_args: dict, output_args: dict):
    output_args['c'] = 'copy'
    output_args['c:s'] = 'mov_text'
    output_args['metadata:s:s:0'] = f'language={transcribed.subtitles.language}'

    input_stream = ffmpeg.input(input_path, **input_args)
    subtitle_stream = ffmpeg.input(transcribed.tmp_file_path)

    if translated.tmp_file is None:
        ffmpeg.output(
            input_stream, subtitle_stream, output_path, **output_args
        ).run(quiet=True, overwrite_output=True)
    else:
        output_args['metadata:s:s:1'] = f'language={translated.subtitles.language}'
        translated_stream = ffmpeg.input(translated.tmp_file_path)
        ffmpeg.output(
            input_stream, subtitle_stream, translated_stream, output_path, **output_args
        ).run(quiet=True, overwrite_output=True)
