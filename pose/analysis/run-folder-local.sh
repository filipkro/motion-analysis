#!/bin/bash

TMPDIR="/home/filipkr/Documents/xjob"
### TODO: fix file structure and script to automatically change config and checkpoint
### TODO: automate detection of all videos
MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth"
MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py"
VIDEO_FOLDER="$TMPDIR/vids/folder-test"
FOLDER_BOX="$TMPDIR/motion-analysis/pose/mmpose/mmdetection/"
OUT_DIR="$TMPDIR/vids/out/folder-test"
FILE_NAME=""
ONLY_BOX=false

python analyse_folder.py $MODEL_CONFIG $MODEL_CHECKPOINT $VIDEO_FOLDER --out-video-root $OUT_DIR --folder_box $FOLDER_BOX --show true