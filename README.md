# WP4 End-to-End Driving using Nvidia cameras

This repository contains code to train end-to-end model using Rally Estonia 2020 dataset. Only steering angle
is predicted using front wide camera. Throttle is not predicted and must be controlled using other means.

## Dataset

https://docs.google.com/spreadsheets/d/1AaAbLjStrIYLI6l3RYshKFQz80Ov_siAtBU5WWGc8ew/edit#gid=0

## Training


## Models

*models* directory contains pretrained models. All models are trained until validation loss stops to improve for 10 epochs.

*best.pt* is model with the smallest validation loss, *best.onxx* is the same model saved into ONNX format. *last.pt* is model
saved after last training epoch.

- *wide-v1* - middle wide angle camera (A0) trained on only Sulaoja track
- *wide-v2* - middle wide angle camera (A0) trained on all tracks
- *wide-aug-v1* - middle wide angle camera (A0) with heuristic augmentation from side cameras (A1, A2) trained on only Sulaoja track
- *wide-aug-v2* - middle wide angle camera (A0) with Stanley-like augmentation from side cameras (A1, A2) trained on only Sulaoja track
- *wide-aug-v3* - middle wide angle camera (A0) with Stanley-like augmentation from side cameras (A1, A2) trained on all tracks