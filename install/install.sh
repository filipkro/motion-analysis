pythonpath="$(which python)"
if [$# -eq 0]
then
    pip install -r requirements.txt
    pip install -v -e pose/mmpose/mmdetection/

else
    pip install -r requirements-cluster.txt
    pip install -v -e pose/mmpose/mmdetection/ --prefix $pythonpath

fi


#install mmpose:
cd ../pose/mmpose
python setup.py develop
cd ../..
