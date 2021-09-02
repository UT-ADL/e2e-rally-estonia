# WP4 End-to-End Driving Dataset

## Bag files

Recorded bags are stored on the UT Rocket HPC (rocket.hpc.ut.ee) in `/gpfs/space/projects/Bolt/bagfiles`. More information
about the bag files can be found in [spreadsheet](https://docs.google.com/spreadsheets/d/1AaAbLjStrIYLI6l3RYshKFQz80Ov_siAtBU5WWGc8ew/edit#gid=0).

## Environment

To extract bags, local ROS must be installed, which is easiest to do using:

```bash
conda create -n ros ros-noetic-desktop pandas -c conda-forge -c robostack && conda activate ros
```

## Extracting

Dataset can be extracted from bags running jupyter notebook `data_extract/extract_images.ipynb` or using 
`data_extract/extract.sh` bash script: 

```bash
cd data_extract
./extract.sh datasets/ut/nvidia-data/2021-06-21-14-31-56_e2e_vahi_back_nvidia_wide-v2_11.bag
```

Dataset is extracted into the same directory where the bag file resides.