#!/bin/bash

echo "Bag file: $1"

python3 ./image_extractor.py --extract-lidar --bag-file=/gpfs/space/projects/Bolt/bagfiles/$1 --extract-dir=$2