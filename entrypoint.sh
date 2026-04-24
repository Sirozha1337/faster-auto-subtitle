#!/bin/bash
NVIDIA_PATH=$(python -c "import nvidia; print(nvidia.__path__[0])" 2>/dev/null)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_PATH/cudnn/lib
faster_auto_subtitle $@ --output_dir /app/output /app/input