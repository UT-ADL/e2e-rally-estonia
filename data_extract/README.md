# WP4 End-to-End Driving Dataset

## Bag files

Recorded bags are stored on the UT Rocket HPC (rocket.hpc.ut.ee) in `/gpfs/space/projects/Bolt/bagfiles`. More information
about the bag files can be found in [spreadsheet](https://docs.google.com/spreadsheets/d/1AaAbLjStrIYLI6l3RYshKFQz80Ov_siAtBU5WWGc8ew/edit#gid=0).

## Environment

To extract bags, local ROS must be installed, which is easiest to do using:

```bash
conda create -n ros ros-noetic-desktop pandas tqdm jupyter -c conda-forge -c robostack && conda activate ros
```

See more at https://medium.com/robostack/cross-platform-conda-packages-for-ros-fa1974fd1de3

## Extracting

Dataset can be extracted from bags running `data_extract/extract.sh` bash script: 

```bash
cd data_extract
python ./image_extractor.py --bag-file=/media/romet/data2/datasets/rally-estonia/bags/2021-05-28-15-19-48_e2e_sulaoja_20_30.bag --extract-dir=/media/romet/data2/datasets/rally-estonia/test-dataset
```
Check the script for additional arguments.

## HPC

Dataset can be re-extracted in Rocket HPC by checking out this repository and running _data_extract/extract_all.job_ using sbatch:

```bash
cd nvidia-e2e/data_extract
sbatch extract_all.job
```

All bags are extracted in parallel and it should take about half a day to finish.