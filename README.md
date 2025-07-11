# Automatic subtitles in your videos

This is a fork of [auto_subtitle](https://github.com/m1guelpf/auto-subtitle)
using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementation.

This repository uses `ffmpeg` and [OpenAI's Whisper](https://openai.com/blog/whisper) to automatically generate and
overlay subtitles on any video. 

It can also use [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) or [deep translator](https://pypi.org/project/deep-translator/) to translate subtitles to another language.

While both transcription and translation (if using Opus-MT) are offline processes they require downloading pre-trained models that require some time to load on the first run. 

## Installation

### Docker Container (recommended)

This method is recommended for most users as it isolates dependencies and ensures compatibility.

You can download the latest Docker image from GitHub Container Registry:

    docker pull ghcr.io/sirozha1337/faster-auto-subtitle:latest

Then run the container:

    docker run --gpus all --rm -v /path/to/cache:/root/.cache/huggingface/hub -v /path/to/video.mp4:/app/input/video.mp4 -v /path/to/output:/app/output ghcr.io/sirozha1337/faster-auto-subtitle:latest

Remember to replace `/path/to/video.mp4` with the path to your video file and `/path/to/output` with the path where you want the output files to be saved. 

Also replace `/path/to/cache` with the host directory where you want to cache the Hugging Face models. This will speed up subsequent runs as it would not require downloading those models again.

### Python

#### Requirements

To get started, you'll need Python 3.11 or newer.

You'll also need to install [`ffmpeg`](https://ffmpeg.org/), which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg
```

Newer version of [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) requires installation of [CUDA 12](https://developer.nvidia.com/cuda-downloads) to run on GPU, or you can run it on CPU with `--device cpu` option.

Additional CUDA installation instructions can be found [here](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#gpu).

#### Virtual Environment (recommended)

To isolate dependencies, you can use Python’s built-in venv module:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install git+https://github.com/Sirozha1337/faster-auto-subtitle.git
    
    # For Thai language translation support, run this command:
    pip install thai-segmenter==0.4.2

#### System-wide Installation

Install the binary by running the following command:

    pip install git+https://github.com/Sirozha1337/faster-auto-subtitle.git

    # In case you need to translate Thai language
    pip install thai-segmenter==0.4.2

## Usage

For the sake of conciseness, the usage is given for the command-line interface (CLI) version of the tool. 
If you're using a Docker container, you can use same CLI arguments just omitting the output and input file paths, as they are already set to the mounted volumes in the container.

The following command will generate a `subtitled/video.mp4` file contained the input video with overlayed subtitles.

    faster_auto_subtitle /path/to/video.mp4 -o subtitled/

You can also specify a folder with multiple videos, and it will process all of them:

    faster_auto_subtitle /path/to/videos/ -o subtitled/

The default setting (which selects the `small` model) works well for transcribing English. You can optionally use a
bigger model for better results (especially with other languages). The available models
are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `large-v1`, `large-v2`, `large-v3`.

    faster_auto_subtitle /path/to/video.mp4 --model medium

Adding `--task translate` will translate the subtitles into English:

    faster_auto_subtitle /path/to/video.mp4 --task translate

Adding `--target_language {2-letter-language-code}` will translate the subtitles into specified language
using [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) (by default) or [deep translator](https://pypi.org/project/deep-translator/):

    faster_auto_subtitle /path/to/video.mp4 --target_language fr

This will require downloading the appropriate model. If direct translation is not available it will attempt translation
from source to english and from english to source.

Run the following to view all available options:

    faster_auto_subtitle --help

## Deep-Translator Translation Mode

You can use the deep-translator package for subtitle translation with various backends (Google, Deepl, MyMemory, etc.).

### Example: Google Translate (no API key required)

```sh
faster_auto_subtitle your_video.mp4 \
  --task translate \
  --translator_mode deep-translator \
  --deep_translator_backend google
```

### Example: Deepl (API key required)

```sh
faster_auto_subtitle your_video.mp4 \
  --task translate \
  --translator_mode deep-translator \
  --deep_translator_backend deepl \
  --deep_translator_kwargs '{"api_key": "YOUR_DEEPL_API_KEY"}'
```

### Supported Backends

- google
- mymemory
- deepl
- qcri
- linguee
- pons
- yandex
- microsoft
- papago
- libre
- baidu

For more, see the [deep-translator documentation](https://deep-translator.readthedocs.io/).


## Tips

The tool also exposes a couple of model parameters, that you can tweak to increase accuracy.

Higher `beam_size` usually leads to greater accuracy, but slows down the process.

Setting higher `no_speech_threshold` could be useful for videos with a lot of background noise to stop Whisper from "
hallucinating" subtitles for it.

You can use `sample_interval` parameter to generate subtitles for a portion of the video to play around with those
parameters:

    faster_auto_subtitle /path/to/video.mp4 --model medium --sample_interval 00:05:30-00:07:00 --beam_size 6 --no_speech_threshold 0.7

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
