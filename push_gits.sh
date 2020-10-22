git add .
git commit -m"$1"
git push
git -C pose/mmpose add .
git -C pose/mmpose commit -m"$1"
git -C pose/mmpose push
git -C pose/mmpose/mmdetection add .
git -C pose/mmpose/mmdetection commit -m"$1"
git -C pose/mmpose/mmdetection push
git -C tsc/dl-4-tsc add .
git -C tsc/dl-4-tsc commit -m"$1"
git -C tsc/dl-4-tsc push
