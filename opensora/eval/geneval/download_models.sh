#!/bin/bash

# Download Mask2Former object detection config and weights

if [ ! -z "$1" ]
then
    mkdir -p "$1"
    wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "$1/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
fi
