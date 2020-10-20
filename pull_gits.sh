# TODO: fix with -C flag and something workingfor git 1.8...
dir="$(pwd)"
git pull
cd pose/mmpose
git pull
cd mmdetection
git pull
cd $dir
cd tsc/dl-4-tsc
git pull
cd $dir
