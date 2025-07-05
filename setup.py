from setuptools import setup, find_packages

setup(
    version="1.5",
    name="faster_auto_subtitle",
    packages=find_packages(),
    py_modules=["faster_auto_subtitle"],
    author="Sergey Chernyaev",
    install_requires=[
        'faster-whisper==1.1.1',
        'tqdm==4.66.4',
        'ffmpeg-python==0.2.0',
        'langcodes==3.3.0',
        'transformers==4.53.1',
        'torch==2.7.1',
        'numpy==2.3.1',
        'nltk==3.9.1',
        'huggingface_hub==0.33.2',
        'sentencepiece==0.2.0',
        'sacremoses==0.1.1',
    ],
    description="Automatically generate and embed subtitles into your videos",
    entry_points={
        'console_scripts': ['faster_auto_subtitle=faster_auto_subtitle.cli:main'],
    },
    include_package_data=True,
)
