#!/bin/bash

echo "Bag file: $1"

cp /gpfs/space/projects/Bolt/bagfiles/$1 /tmp/$1
python3 ./image_extractor.py --bag-file=/tmp/$1 --extract-dir=$2