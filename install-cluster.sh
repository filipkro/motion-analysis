pip install torchvision==0.5.0 # make sure compatible with pytorch on cluster
pip install mmcv-full

pip install -r requirements-cluster.txt

cd pose/mmpose/mmdetection
pip install -v -e .

cd ..

#install mmpose:
python setup.py develop

cd ../..
