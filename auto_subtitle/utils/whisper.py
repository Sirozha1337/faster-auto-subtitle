import warnings
import faster_whisper
from tqdm import tqdm


class WhisperAI:
    """
    Wrapper class for the Whisper speech recognition model with additional functionality.

    This class provides a high-level interface for transcribing audio files using the Whisper
    speech recognition model. It encapsulates the model instantiation and transcription process,
    allowing users to easily transcribe audio files and iterate over the resulting segments.

    Usage:
    ```python
    whisper = WhisperAI(model_args, transcribe_args)

    # Transcribe an audio file and iterate over the segments
    for segment in whisper.transcribe(audio_path):
        # Process each transcription segment
        print(segment)
    ```

    Args:
    - model_args: Arguments to pass to WhisperModel initialize method
        - model_size_or_path (str): The name of the Whisper model to use.
        - device (str): The device to use for computation ("cpu", "cuda", "auto").
        - compute_type (str): The type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
    - transcribe_args (dict): Additional arguments to pass to the transcribe method.

    Attributes:
    - model (faster_whisper.WhisperModel): The underlying Whisper speech recognition model.
    - transcribe_args (dict): Additional arguments used for transcribe method.

    Methods:
    - transcribe(audio_path): Transcribes an audio file and yields the resulting segments.
    """

    def __init__(self, model_args: dict, transcribe_args: dict):
        self.model = faster_whisper.WhisperModel(**model_args)
        self.transcribe_args = transcribe_args

    def transcribe(self, audio_path: str):
        """
        Transcribes the specified audio file and yields the resulting segments.

        Args:
        - audio_path (str): The path to the audio file for transcription.

        Yields:
        - faster_whisper.TranscriptionSegment: An individual transcription segment.
        """
        warnings.filterwarnings("ignore")
        segments, info = self.model.transcribe(
            audio_path, **self.transcribe_args)
        warnings.filterwarnings("default")

        return (self.subtitles_iterator(segments, info), info)

    def subtitles_iterator(self, segments, info):
        # Same precision as the Whisper timestamps.
        total_duration = round(info.duration, 2)

        with tqdm(total=total_duration, unit=" seconds") as pbar:
            for segment in segments:
                yield segment
                pbar.update(segment.end - segment.start)
            pbar.update(0)
