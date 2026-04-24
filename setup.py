from setuptools import setup, find_packages

setup(
    version="1.9.0",
    name="faster_auto_subtitle",
    packages=find_packages(),
    py_modules=["faster_auto_subtitle"],
    author="Sergey Chernyaev",
    install_requires=[
        'faster-whisper==1.2.1',
        'tqdm==4.66.4',
        'ffmpeg-python==0.2.0',
        'langcodes==3.5.1',
        'transformers==5.6.2',
        'torch==2.11.0',
        'numpy==2.4.4',
        'nltk==3.9.4',
        'huggingface_hub==1.11.0',
        'sentencepiece==0.2.1',
        'sacremoses==0.1.1',
        'deep-translator==1.11.4',
    ],
    description="Automatically generate and embed subtitles into your videos",
    entry_points={
        'console_scripts': ['faster_auto_subtitle=faster_auto_subtitle.cli:main'],
    },
    include_package_data=True,
)
