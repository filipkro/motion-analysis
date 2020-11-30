pip install torchvision==0.7.0 # make sure compatible with pytorch on cluster
pip install mmcv-full==1.1.5

pip install -r install/requirements-cluster.txt

cd pose/mmpose/mmdetection
pip install -v -e .

cd ..

#install mmpose:
python setup-cluster.py develop

cd ../..
