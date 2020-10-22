#!/bin/bash

### TODO: fix file structure and script to automatically change config and checkpoint
### TODO: automate detection of all videos
#MODEL_CHECKPOINT="/home/filipkr/Documents/xjob/mmpose/checkpoints/top-down/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth"
#MODEL_CONFIG="/home/filipkr/Documents/xjob/mmpose/configs/top_down/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py"
FOLDER_2D="$TMPDIR/motion-analysis/pose/mmpose/mmdetection/"
VIDEO="$TMPDIR/025_FL_R.MOV"
OUT_DIR="$TMPDIR/results/"
FILE_NAME=""
ONLY_BOX=false

python anlyse_vid.py $MODEL_CONFIG $MODEL_CHECKPOINT --video-path $VIDEO --out-video-root $OUT_DIR --on_cluster 1 --device cuda:0 --folder_2d $FOLDER_2D
