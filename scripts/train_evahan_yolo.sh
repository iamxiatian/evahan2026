#!/bin/bash
python train.py --data evahan \
--model m-evahan \
--epoch 500 \
--image-size 1120 \
--batch-size 32 \
--project public_dataset/evahan_20260206_lr0001_imgsz1120_Adam_addtextcolor \
--plot 1 \
--save-period 2 \
--device 3,5,6,7 \
--workers 6 \
--optimizer Adam \
--lr0 0.001 \
--mosaic 0.4 \
--pretrain local/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt
