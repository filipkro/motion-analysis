if [$# -eq 0]
then
    pip install -r requirements.txt
else
    pip install -r requirements-cluster.txt
fi

pip install -v -e pose/mmpose/mmdetection/

#install mmpose:
cd pose/mmpose
python setup.py develop
cd ...
