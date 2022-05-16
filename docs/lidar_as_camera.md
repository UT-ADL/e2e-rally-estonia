# LiDAR-as-Camera for End-to-End Driving

This repository contains the code to reproduce the experiments of the paper "LiDAR-as-Camera for End-to-End Driving".
All used models can be trained and off-policy metrics reproduced. 

Task list:
- [ ] training models
- [x] calculating off-policy metrics
- [ ] calculating on-policy metrics

## Training models

### Camera v1, v2, v3, overfit

### LiDAR v1, v2, v3, overfit

## Off-policy metrics

Off-policy metrics can be calculated by running following script from root folder: 

```bash
python -m metrics.calculate_model_ol_metrics.py --root-path <path to extracted dataset>
```

Use `--root-path` parameter to defined path where Rally Estonia Dataset is downloaded and extracted.

## On-policy metrics

