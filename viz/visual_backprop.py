import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from dataloading.nvidia import NvidiaDataset
from dataloading.ouster import OusterDataset
from network import PilotNet, PilotNetOld


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
        model = PilotNet()
    else:
        print(f"Unknown model type '{args.model_type}'")
        sys.exit()

    model.load_state_dict(torch.load(f"models/{args.model_name}/best.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    deq = deque(range(0, len(dataset)))
    example = dataset[deq[0]][0]["image"].to(device)
    vis = getImagesFull(model, example)[0]

    cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    window_scale = 3
    cv2.resizeWindow('image', window_scale*68, window_scale*264)

    while cv2.getWindowProperty('vis', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.imshow('vis', vis)
        k = cv2.waitKey(10)
        if k == ord('j'):
            deq.rotate(1)
            example = dataset[deq[0]][0]["image"].to(device)
            vis = getImagesFull(model, example)[0]
        elif k == ord('k'):
            deq.rotate(-1)
            example = dataset[deq[0]][0]["image"].to(device)
            vis = getImagesFull(model, example)[0]

