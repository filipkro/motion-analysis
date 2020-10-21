pythonpath="$(which python)"

pip install -r requirements-cluster.txt
pip install -v -e pose/mmpose/mmdetection/ #--prefix $pythonpath

#install mmpose:
cd pose/mmpose
python setup.py develop
cd ../analysis
