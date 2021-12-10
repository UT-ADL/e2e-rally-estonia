echo "Creating video for dataset: $1"

python3 ./video_creator.py --bag-file=/gpfs/space/projects/Bolt/dataset/$1 --video-type driving