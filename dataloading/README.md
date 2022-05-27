# Data loading

## Download dataset

Dataset is extracted in Rocket HPC and is located in directory _/gpfs/space/projects/Bolt/dataset_. It is quite large
and is best synced into training environment using rsync:

```bash
rsync -ah --info=progress2 --exclude '*/left' --exclude '*/right' <username>@rocket.hpc.ut.ee:/gpfs/space/projects/Bolt/dataset-new-small .
```
