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
    whisper = WhisperAI(model_name, device, compute_type, model_args)

    # Transcribe an audio file and iterate over the segments
    for segment in whisper.transcribe(audio_path):
        # Process each transcription segment
        print(segment)
    ```

    Args:
    - model_name (str): The name of the Whisper model to use.
    - device (str): The device to use for computation ("cpu", "cuda", "auto").
    - compute_type (str): The type to use for computation.
        See https://opennmt.net/CTranslate2/quantization.html.
    - model_args (dict): Additional arguments to pass to the transcribe method.

    Attributes:
    - model (faster_whisper.WhisperModel): The underlying Whisper speech recognition model.
    - model_args (dict): Additional arguments used for transcribe method.

    Methods:
    - transcribe(audio_path): Transcribes an audio file and yields the resulting segments.
    """

    def __init__(self, model_name, device, compute_type, model_args):
        self.model = faster_whisper.WhisperModel(
            model_name, device=device, compute_type=compute_type)
        self.model_args = model_args

    def transcribe(self, audio_path):
        warnings.filterwarnings("ignore")
        segments, info = self.model.transcribe(audio_path, **self.model_args)
        warnings.filterwarnings("default")

        total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.

        with tqdm(total=total_duration, unit=" seconds") as pbar:
            for segment in segments:
                yield segment
                pbar.update(segment.end - segment.start)
            pbar.update(0)
