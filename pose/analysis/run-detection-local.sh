#!/bin/bash

TMPDIR="/home/filipkr/Documents/xjob"
### TODO: fix file structure and script to automatically change config and checkpoint
### TODO: automate detection of all videos
MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth"
MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py"
#MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_256x192_dark-8cba3197_20200812.pth"
#MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_256x192_dark.py"
FOLDER_BOX="$TMPDIR/motion-analysis/pose/mmpose/mmdetection/"
VIDEO="$1"
OUT_DIR="$TMPDIR/vids/out/"
FILE_NAME=""
ONLY_BOX=false
FLIP2RIGHT=true
FNAME_FORMAT=true
SKIP_RATE=1

python analyse_vid.py $MODEL_CONFIG $MODEL_CHECKPOINT --video-path $VIDEO --out-video-root $OUT_DIR --folder_box $FOLDER_BOX --show true --flip2right $FLIP2RIGHT --fname_format $FNAME_FORMAT --skip_rate $SKIP_RATE
