import os
import tempfile
import logging
from typing import Optional
import ffmpeg
from .tempfile import SubtitlesTempFile
from .files import filename
from ..models.Subtitles import Subtitles

logger = logging.getLogger(__name__)


def get_audio(path: str, audio_channel_index: int, sample_interval: Optional[list] = None) -> str:
    temp_dir = tempfile.gettempdir()

    file_name = filename(path)
    logger.info("Extracting audio from %s...", file_name)
    output_path = os.path.join(temp_dir, f"{file_name}.wav")

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


def file_has_audio(path: str) -> bool:
    try:
        audio_info = ffmpeg.probe(path, select_streams='a')
        return 'streams' in audio_info \
            and audio_info['streams'] is not None \
            and len(audio_info['streams']) > 0
    except ffmpeg.Error:
        return False


def preprocess_audio(path: str, audio_channel_index: int, sample_interval: Optional[list]) -> str:
    if sample_interval is not None or audio_channel_index != 0:
        return get_audio(path, audio_channel_index, sample_interval)

    audio_info = ffmpeg.probe(path, select_streams='a')
    audio_format = audio_info['format']
    audio_streams = audio_info['streams']
    if audio_format['format_name'] == 'wav' and \
            audio_streams is not None and len(audio_streams) == 1:
        audio_stream = audio_streams[0]
        if audio_stream['codec_name'] == 'pcm_s16le' and audio_stream['sample_rate'] == '16000':
            return path

    return get_audio(path, audio_channel_index)


def add_subtitles(path: str, transcribed: Subtitles, translated: Optional[Subtitles],
                  sample_interval: list, output_args: dict[str, str]) -> None:
    file_name = filename(path)
    out_path = os.path.join(output_args["output_dir"], f"{file_name}.mp4")

    logger.info("Adding subtitles to %s...", file_name)

    ffmpeg_input_args = {}
    if sample_interval is not None:
        ffmpeg_input_args['ss'] = str(sample_interval[0])

    ffmpeg_output_args = {}
    if sample_interval is not None:
        ffmpeg_output_args['t'] = str(
            sample_interval[1] - sample_interval[0])

    # HACK: On Windows it's impossible to use absolute subtitle file path with ffmpeg,
    # so we use temp copy instead
    # see: https://github.com/kkroening/ffmpeg-python/issues/745
    with SubtitlesTempFile(transcribed) as transcribed_tmp, SubtitlesTempFile(
            translated) as translated_tmp:

        if output_args["subtitle_type"] == 'hard':
            hard_subtitles(path, out_path, transcribed_tmp, translated_tmp, ffmpeg_input_args,
                           ffmpeg_output_args)
        elif output_args["subtitle_type"] == 'soft':
            soft_subtitles(path, out_path, transcribed_tmp, translated_tmp, ffmpeg_input_args,
                           ffmpeg_output_args)

    logger.info("Saved subtitled video to %s.", os.path.abspath(out_path))


def hard_subtitles(input_path: str, output_path: str,
                   transcribed: SubtitlesTempFile, translated: SubtitlesTempFile,
                   input_args: dict, output_args: dict) -> None:
    video = ffmpeg.input(input_path, **input_args)
    audio = video.audio

    intermediate = video.filter(
        'subtitles', transcribed.tmp_file_path,
        force_style="OutlineColour=&H40000000,BorderStyle=3")

    if translated.tmp_file is not None:
        intermediate = intermediate.filter(
            'subtitles', translated.tmp_file_path,
            force_style="OutlineColour=&H40000000,BorderStyle=3,Alignment=6")
    ffmpeg.concat(
        intermediate, audio, v=1, a=1
    ).output(output_path, **output_args) \
        .run(quiet=True, overwrite_output=True)


def soft_subtitles(input_path: str, output_path: str,
                   transcribed: SubtitlesTempFile, translated: SubtitlesTempFile,
                   input_args: dict, output_args: dict) -> None:
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
