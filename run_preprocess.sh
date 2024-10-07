#!/bin/bash

# Usage: sh run_preprocess.sh <filelist>

if [ -z "$1" ]; then
  echo "Error: No filelist argument supplied."
  echo "Usage: ./run_preprocess.sh <filelist>"
  exit 1
fi

FILELIST=$1
OUTPUT_FEATURE_PATH="feat.bin"

python preprocess.py --filelist "$FILELIST" --output_feature_path "$OUTPUT_FEATURE_PATH"