#!/bin/bash

TMPDIR="/home/filipkr/Documents/xjob"
### TODO: fix file structure and script to automatically change config and checkpoint
### TODO: automate detection of all videos
MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth"
MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py"
FOLDER_BOX="$TMPDIR/motion-analysis/pose/mmpose/mmdetection/"
VIDEO="$TMPDIR/vids/real/vids-before-after/012/Baseline/012_FL_R.mts"
OUT_DIR="$TMPDIR/vids/out/"
FILE_NAME=""
ONLY_BOX=false

python analyse_vid.py $MODEL_CONFIG $MODEL_CHECKPOINT --video-path $VIDEO --out-video-root $OUT_DIR --folder_box $FOLDER_BOX --show true
