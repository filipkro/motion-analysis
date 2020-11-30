#!/bin/bash

### TODO: fix file structure and script to automatically change config and checkpoint
### TODO: automate detection of all videos
MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth"
MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py"
VIDEO_FOLDER="$TMPDIR/$1"
FOLDER_BOX="$TMPDIR/motion-analysis/pose/mmpose/mmdetection/"
OUT_DIR="$TMPDIR/results"
ONLY_BOX=false
FLIP2RIGHT=true
FNAME_FORMAT=true

echo "launching python scripts"

echo "$(which python)"
echo "$(ls)"
python analyse_folder.py $MODEL_CONFIG $MODEL_CHECKPOINT $VIDEO_FOLDER --out-video-root $OUT_DIR --device cuda:0 --folder_box $FOLDER_BOX --show false --save_pixels false --save4_3d true --allow_flip false --flip2right $FLIP2RIGHT --fname_format $FNAME_FORMAT 
