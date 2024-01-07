pip install torchvision==0.7.0 # make sure compatible with pytorch on cluster
#pip install numpy==
pip install mmcv-full==1.1.5
pip install pytz>=2017.2
#pip install mmpycocotools

pip install -r install/requirements-cluster.txt

cd pose/mmpose/mmdetection
pip install -v -e .

cd ..

#install mmpose:
python setup-cluster.py develop
pip install numpy==1.17.3
cd ../..
