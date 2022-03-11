import argparse
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from dataloading.nvidia import NvidiaDataset
from dataloading.ouster import OusterDataset
from network import PilotNet, PilotNetOld

"""
Adapted from:
https://github.com/javangent/ouster-e2e/blob/4bfafaf764de85f87ac4a4d71d21fbf9a333790f/visual_backprop.py
https://github.com/mbojarski/VisualBackProp/blob/master/vis.lua
"""
def outer_hook(activations):
    def hook(module, inp, out):
        activations.append(out)

    return hook


def findModules(model, layer_str):
    modules = []
    for layer in model.children():
        if layer_str in str(layer):
            modules.append(layer)
    return modules


def registerHooks(model, layer_str, activations):
    handles = []
    for i, layer in enumerate(model.children()):
        if layer_str in str(layer):
            handle = layer.register_forward_hook(outer_hook(activations))
            handles.append(handle)
    return handles


def removeHandles(handles):
    for handle in handles:
        handle.remove()


def calculateAdj(targetSize, ker, pad, stride):
    out = []
    for i in range(len(targetSize)):
        out.append((targetSize[i] + 2 * pad[i] - ker[i]) % stride[i])
    return tuple(out)


def normalizeBatch(out):
    height, width = out.shape[-2:]
    out = out.view(out.shape[0], out.shape[1], -1)
    out -= out.min(2, keepdim=True)[0]
    out /= out.max(2, keepdim=True)[0]
    out = out.view(out.shape[0], out.shape[1], height, width)


def getVisMask(mod, idata):
    with torch.no_grad():
        activations = []
        handles = registerHooks(mod.features, 'ReLU', activations)

        # do the forward pass through the feature extractor (convolutional layers)
        mod(idata)
        removeHandles(handles)

        del handles

        layersConv = findModules(mod.features, 'Conv2d')
        # mask = None
        sumList = [None] * len(layersConv)
        sumListUp = [None] * len(layersConv)
        fMaps = [None] * len(layersConv)
        fMapsMasked = [None] * len(layersConv)
        # process feature maps
        for i in reversed(range(len(layersConv))):
            # sum all the feature maps at each level
            sumList[i] = activations[i].sum(-3, keepdim=True)  # channel-wise
            # calculate the dimension of scaled up map
            fMaps[i] = sumList[i]
            # pointwise multiplication
            if i < len(layersConv) - 1:
                sumList[i] *= sumListUp[i + 1]

            # save intermediate mask
            fMapsMasked[i] = sumList[i]
            # scale up intermediate mask using deconvolution
            if i > 0:
                inp_shape = activations[i - 1].shape[-2:]
            else:
                inp_dhape = idata.shape[-2:]

            output_padding = calculateAdj(inp_shape,
                                          layersConv[i].kernel_size,
                                          layersConv[i].padding,
                                          layersConv[i].stride)

            mmUp = nn.ConvTranspose2d(1, 1,
                                      layersConv[i].kernel_size,
                                      layersConv[i].stride,
                                      layersConv[i].padding,
                                      output_padding)

            mmUp.cuda()
            torch.nn.init.zeros_(mmUp.bias)
            torch.nn.init.ones_(mmUp.weight)
            sumListUp[i] = mmUp(sumList[i])

        # assign output - visualization mask
        out = sumListUp[0]
        # normalize mask to range 0-1
        normalizeBatch(out)

        # return visualization mask, averaged feature maps, and intermediate masks
        return out, fMaps, fMapsMasked


def getImages(imgBatch, visMask, fMaps, fMapsM):
    b, c, h, w = visMask.shape
    imgOut = torch.zeros_like(imgBatch)
    spacing = 2
    input_img_channel = 1
    fMapsImg = torch.ones(b, c, len(fMaps) * h + (len(fMaps) - 1) * spacing, w).cuda()
    fMapsImgM = torch.ones(b, c, len(fMaps) * h + (len(fMaps) - 1) * spacing, w).cuda()
    # normalize and scale averaged feature maps and intermediate visualization masks
    for i in range(len(fMaps)):
        normalizeBatch(fMaps[i])
        normalizeBatch(fMapsM[i])
        offset_h = i * (h + spacing)
        fMapsImg[:, :, offset_h:offset_h + h] = F.resize(fMaps[i], (h, w)).cuda()
        fMapsImgM[:, :, offset_h:offset_h + h] = F.resize(fMapsM[i], (h, w)).cuda()

    # overlay visualization mask over the input images
    imgOut[:, 0] = imgBatch[:, input_img_channel] - visMask[:, 0]
    imgOut[:, 1] = imgBatch[:, input_img_channel] + visMask[:, 0]
    imgOut[:, 2] = imgBatch[:, input_img_channel] - visMask[:, 0]
    imgOut = imgOut.clamp(0, 1)
    return imgOut, fMapsImg, fMapsImgM


def getImagesFull(model, example):
    # example = example[:, :, 256:-256].unsqueeze(0) # create batch dimension
    example = example[:, :, :].unsqueeze(0)  # create batch dimension
    out, fMaps, fMapsMasked = getVisMask(model, example)
    imgOut, fMapsImg, fMapsImgM = getImages(example, out, fMaps, fMapsMasked)
    imgOut = imgOut[0].permute(1, 2, 0).cpu().numpy()
    fMapsImg = fMapsImg[0].permute(1, 2, 0).cpu().numpy()
    fMapsImgM = fMapsImgM[0].permute(1, 2, 0).cpu().numpy()
    return imgOut, fMapsImg, fMapsImgM


def draw_steering_angle(frame, steering_angle, steering_wheel_radius, steering_position, size, color):
    steering_angle_rad = math.radians(steering_angle)
    x = steering_wheel_radius * np.cos(np.pi / 2 + steering_angle_rad)
    y = steering_wheel_radius * np.sin(np.pi / 2 + steering_angle_rad)
    cv2.circle(frame, (steering_position[0] + int(x), steering_position[1] - int(y)), size, color, thickness=-1)

def getImageWithOverlay(model, frame):
    example = frame["image"].to(device)
    img = example.cpu().permute(1, 2, 0).detach().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vis = getImagesFull(model, example)[0]
    result = cv2.vconcat([img, vis])

    scale_percent = 500  # percent of original size
    width = int(result.shape[1] * scale_percent / 100)
    height = int(result.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

    steering_angle = math.degrees(frame["steering_angle"])
    vehicle_speed = frame["vehicle_speed"]
    turn_signal = int(frame["turn_signal"])
    cv2.putText(resized, 'True: {:.2f} deg, {:.2f} km/h'.format(steering_angle, vehicle_speed), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    pred = model(example.unsqueeze(0)).squeeze(1).cpu().detach().numpy()[0]
    if len(pred) == 1:
        pred_steering_angle = math.degrees(pred[0])
    elif len(pred) == 3:
        pred_steering_angle = math.degrees(pred[turn_signal])
    else:
        print(f"Unknown prediction size: {len(pred)}")
        sys.exit()

    cv2.putText(resized, 'Pred: {:.2f} deg'.format(pred_steering_angle), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)

    turn_signal_map = {
        1: "straight",
        2: "left",
        0: "right"
    }
    cv2.putText(resized, 'turn signal: {}'.format(turn_signal_map.get(turn_signal, "unknown")), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # draw steering wheel
    radius = 100
    steering_pos = (150, 270)
    cv2.circle(resized, steering_pos, radius, (255, 255, 255), 7)
    draw_steering_angle(resized, steering_angle, radius, steering_pos, 13, (0, 255, 0))
    draw_steering_angle(resized, pred_steering_angle, radius, steering_pos, 9, (0, 0, 255))


    return resized


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model used for predictions.'
    )

    argparser.add_argument(
        '--model-type',
        required=False,
        default="pilotnet",
        choices=['pilotnet', 'pilotnet-old'],
    )

    argparser.add_argument(
        '--dataset-name',
        required=True,
        help='Name of the dataset used for predictions.'
    )

    argparser.add_argument(
        '--input-modality',
        required=False,
        default="nvidia-camera",
        choices=['nvidia-camera', 'ouster-lidar'],
    )

    argparser.add_argument(
        '--conditional-learning',
        default=False,
        action='store_true',
        help="When true, network is trained with conditional branches using turn blinkers."
    )

    args = argparser.parse_args()

    root_path = Path("/home/romet/data2/datasets/rally-estonia/dataset-small")
    data_paths = [root_path / args.dataset_name]
    if args.input_modality == "nvidia-camera":
        dataset = NvidiaDataset(data_paths)
    elif args.input_modality == "ouster-lidar":
        dataset = OusterDataset(data_paths)
    else:
        print(f"Uknown input modality '{args.input_modality}'")
        sys.exit()

    if args.model_type == "pilotnet-old":
        model = PilotNetOld()
    elif args.model_type == "pilotnet":
        n_branches = 3 if args.conditional_learning else 1
        model = PilotNet(n_branches=n_branches)
    else:
        print(f"Unknown model type '{args.model_type}'")
        sys.exit()

    model.load_state_dict(torch.load(f"models/{args.model_name}/best.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    deq = deque(range(0, len(dataset)))
    vis = getImageWithOverlay(model, dataset[deq[0]][0])

    cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    window_scale = 500
    cv2.resizeWindow('image', window_scale*2*68, window_scale*264)

    while cv2.getWindowProperty('vis', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.imshow('vis', vis)
        k = cv2.waitKey(10)
        if k == ord('j'):
            deq.rotate(1)
            vis = getImageWithOverlay(model, dataset[deq[0]][0])
        elif k == ord('k'):
            deq.rotate(-1)
            vis = getImageWithOverlay(model, dataset[deq[0]][0])

