#!/bin/bash

echo "Creating video for dataset: $1"

cd ..
python3 ./video_creator.py --dataset-folder=/gpfs/space/projects/Bolt/dataset-paper/$1 --video-type=driving
