#!/bin/bash

VID="$1"
#DIR="$2"
DIR="out"
TMPDIR="/home/filipkr/Documents/xjob"
### TODO: fix file structure and script to automatically change config and checkpoint
MODEL_CHECKPOINT="$TMPDIR/motion-analysis/pose/mmpose/checkpoints/top-down/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
MODEL_CONFIG="$TMPDIR/motion-analysis/pose/mmpose/configs/top_down/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py"
FOLDER_BOX="$TMPDIR/motion-analysis/pose/mmpose/mmdetection/"

if [ ! -d $DIR ]; then
    mkdir $DIR
    echo "directory ${DIR} created"
fi

cp $VID $DIR

#python $TMPDIR/motion-analysis/pose/analysis/analyse_vid.py $MODEL_CONFIG $MODEL_CHECKPOINT --video-path $VID --out-video-root $DIR --folder_box $FOLDER_BOX --show false

cd $DIR

python $TMPDIR/motion-analysis/pose/analysis/utils/extract_reps.py $(pwd) --debug true

python $TMPDIR/motion-analysis/classification/tsc/utils/eval_vid.py $(pwd)
