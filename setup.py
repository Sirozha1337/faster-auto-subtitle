from setuptools import setup, find_packages

setup(
    version="1.2",
    name="faster_auto_subtitle",
    packages=find_packages(),
    py_modules=["auto_subtitle"],
    author="Sergey Chernyaev",
    install_requires=[
        'faster-whisper',
        'tqdm',
        'ffmpeg-python',
        'langcodes',
        'transformers>=4.4,<5',
        'torch>=1.6.0',
        'numpy',
        'nltk',
        'sentencepiece',
        'protobuf',
    ],
    description="Automatically generate and embed subtitles into your videos",
    entry_points={
        'console_scripts': ['faster_auto_subtitle=auto_subtitle.cli:main'],
    },
    include_package_data=True,
)
