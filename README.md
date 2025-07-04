# Automatic subtitles in your videos

This is a fork of [auto_subtitle](https://github.com/m1guelpf/auto-subtitle)
using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementation.

This repository uses `ffmpeg` and [OpenAI's Whisper](https://openai.com/blog/whisper) to automatically generate and
overlay subtitles on any video. 

It also uses [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) to translate subtitles
to another language.

While both transcription and translation are offline processes they require downloading pre-trained models that require 
some time to load on the first run.

## Installation

To get started, you'll need Python 3.9 or newer. Install the binary by running the following command:

    pip install git+https://github.com/Sirozha1337/faster-auto-subtitle.git

    # In case you need to translate Thai language
    pip install thai-segmenter==0.4.2

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

## Usage

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
using [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT):

    faster_auto_subtitle /path/to/video.mp4 --target_language fr

This will require downloading the appropriate model. If direct translation is not available it will attempt translation
from source to english and from english to source.

Run the following to view all available options:

    faster_auto_subtitle --help

## Tips

The tool also exposes a couple of model parameters, that you can tweak to increase accuracy.

Higher `beam_size` usually leads to greater accuracy, but slows down the process.

Setting higher `no_speech_threshold` could be useful for videos with a lot of background noise to stop Whisper from "
hallucinating" subtitles for it.

In my experience settings option `condition_on_previous_text` to `False` dramatically increases accuracy for videos
like TV Shows with an intro song at the start.

You can use `sample_interval` parameter to generate subtitles for a portion of the video to play around with those
parameters:

    faster_auto_subtitle /path/to/video.mp4 --model medium --sample_interval 00:05:30-00:07:00 --condition_on_previous_text False --beam_size 6 --no_speech_threshold 0.7

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
