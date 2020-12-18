git add .
git commit -m"$1"
git push
git -C pose/mmpose add .
git -C pose/mmpose commit -m"$1"
git -C pose/mmpose push
git -C pose/mmpose/mmdetection add .
git -C pose/mmpose/mmdetection commit -m"$1"
git -C pose/mmpose/mmdetection push
git -C classification/tsc add .
git -C classification/tsc commit -m"$1"
git -C classification/tsc push
