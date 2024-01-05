import argparse
from faster_whisper import available_models
from .main import process
from .utils.convert import str2bool, str2timeinterval

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--audio_channel", default="0",
                        type=int, help="audio channel index to use")
    parser.add_argument("--sample_interval", type=str2timeinterval, default=None,
                        help="generate subtitles for a specific fragment of the video (e.g. 01:02:05-01:03:45)")
    parser.add_argument("--model", default="small",
                        choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--device", type=str, default="auto", choices=[
                        "cpu", "cuda", "auto"], help="Device to use for computation (\"cpu\", \"cuda\", \"auto\")")
    parser.add_argument("--compute_type", type=str, default="default", choices=[
                        "int8", "int8_float32", "int8_float16",
                        "int8_bfloat16", "int16", "float16",
                        "bfloat16", "float32"], help="Type to use for computation. See https://opennmt.net/CTranslate2/quantization.html.")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="model parameter, tweak to increase accuracy")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6,
                        help="model parameter, tweak to increase accuracy")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True,
                        help="model parameter, tweak to increase accuracy")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")

    args = parser.parse_args().__dict__

    process(args)


if __name__ == '__main__':
    main()
