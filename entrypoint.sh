#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/
faster_auto_subtitle $@ --output_dir /app/output /app/input