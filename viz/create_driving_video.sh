#!/bin/bash

echo "Creating video for dataset: $1"

cd ..
python3 -m viz.video_creator --dataset-folder=/gpfs/space/projects/Bolt/dataset-demo/$1 --video-type=driving
