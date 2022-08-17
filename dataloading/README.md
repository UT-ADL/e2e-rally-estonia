# Data loading

This package contains implementations of PyTorch dataloaders for using both images from Nvidia cameras and Ouster LiDAR
to train models. Easiest way to get started is to download dataset from UT Rocket HPC. In case access to this is not possible,
access to the bags must be requested and dataset must be extracted following [data_extract/README.md](../data_extract/README.md).

## Download extracted dataset

There is two versions of extracted datasets in Rocket HPC:
 - _/gpfs/space/projects/Bolt/dataset-full_ - camera images saved as _jpeg_ files without being cropped nor resized. Can be 
still used for training models, but with slower performance and preprocessing must be done in the training pipeline. 
This dataset is mostly used to create video from model predictions for analyzing purposes.
 - _/gpfs/space/projects/Bolt/dataset-cropped_ - camera images are cropped and resized to `258x66` used in training to 
improve performance. Images are saved as _png_ files to avoid any loss due to _jpeg_ compression, although there is probably
not much difference and it would make sense to use _jpeg_ instead for even faster training process.

Dataset is extracted in Rocket HPC and is located in directory _/gpfs/space/projects/Bolt/dataset_. It is quite large
and is best synced into training environment using rsync:

```bash
rsync -ah --info=progress2 <username>@rocket.hpc.ut.ee:/gpfs/space/projects/Bolt/dataset-cropped .
```

Use `--exclude '*/left'`, `--exclude '*/right'` and `--exclude '*/lidar'` parameters to exclude side camera and lidar images
respectively to make download size smaller if these inputs are not used.

