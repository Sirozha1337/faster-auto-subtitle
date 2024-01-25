import os
import tempfile
import ffmpeg
from .mytempfile import MyTempFile
from .files import filename, write_srt


def get_audio(paths: list, audio_channel_index: int, sample_interval: list):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg_input_args = {}
        if sample_interval is not None:
            ffmpeg_input_args['ss'] = str(sample_interval[0])

        ffmpeg_output_args = {}
        ffmpeg_output_args['acodec'] = "pcm_s16le"
        ffmpeg_output_args['ac'] = "1"
        ffmpeg_output_args['ar'] = "16k"
        ffmpeg_output_args['map'] = "0:a:" + str(audio_channel_index)
        if sample_interval is not None:
            ffmpeg_output_args['t'] = str(
                sample_interval[1] - sample_interval[0])

        ffmpeg.input(path, **ffmpeg_input_args).output(
            output_path,
            **ffmpeg_output_args
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def overlay_subtitles(subtitles: dict, output_dir: str, sample_interval: list):
    for path, subtitle in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

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
        with MyTempFile(subtitle['output_path'] if 'output_path' in subtitle else None) as srt_temp:
            write_srt(subtitle['segments'], srt_temp.tmp_file)

            video = ffmpeg.input(path, **ffmpeg_input_args)
            audio = video.audio

            ffmpeg.concat(
                video.filter(
                    'subtitles', srt_temp.tmp_file_path,
                    force_style="OutlineColour=&H40000000,BorderStyle=3"), audio, v=1, a=1
            ).output(out_path, **ffmpeg_output_args).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")
