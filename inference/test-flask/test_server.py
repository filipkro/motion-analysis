from flask import Flask
from argparse import Namespace
import sys

sys.path.append('../')
# from inference import main as run_inference
from simple_inference import main as run_inference

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST'])
def predict():
    args = Namespace()
    # args.video = '/home/filipkr/Documents/xjob/vids/real/Videos/MUSSE/musse-SLS/03SLS1R_MUSSE.mts'
    args.video = '/code/03SLS1R_MUSSE.mts'
    args.out = 'out3'
    # run_inference(args)
    run_inference()
    print("in pred")
    return "lol"
