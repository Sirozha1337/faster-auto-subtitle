import warnings
import faster_whisper
from tqdm import tqdm

class WhisperAI:
    def __init__(self, model_name, device, compute_type, model_args):
        self.model = faster_whisper.WhisperModel(model_name, device=device, compute_type=compute_type)
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