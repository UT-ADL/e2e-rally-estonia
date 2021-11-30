#!/bin/bash

echo "Bag file: $1"

python3 ./image_extractor.py --bag-file=$1 --extract-dir=$2 --resize-camera-images