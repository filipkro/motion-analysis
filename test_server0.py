from flask import Flask, request
from argparse import Namespace
import sys
import jobs

# sys.path.append('inference')
# import inference_rq2 as inference_rq
import inference_rq as inference_rq # noqa
# from inference import main as run_inference
# from simple_inference import main as run_inference


print('BEFORE APP \n\n\n\n\n LOL')
app = Flask(__name__)
print('AFTER APP \n\n\n\n\n LOL')
inference_rq.rq.init_app(app)
print('BEFORE INIT_APP \n\n\n\n\n LOL')

jobs.rq.init_app(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!\nLOL1</p>"


@app.route("/task")
def add_task():

    if request.args.get("n"):

        job = jobs.background_task.queue(request.args.get("n"))

        return f"Task ({job.id}) added to queue at {job.enqueued_at}"

    return "No value for count provided"


# @app.route("/predict", methods=['POST'])
# def predict():
#     args = Namespace()
#     print("in pred")
#     args.out = 'out3'
#     job = inference.background_task.queue(args)
#     # run_inference(args)
#
#     if request.args.get("n"):
#
#         job = inference.background_task.queue(request.args.get("n"))
#
#         return f"Prediction started!\nTask ({job.id}) added to queue at {job.enqueued_at}"
#
#     return "No value for count provided"

@app.route("/predict")
def predict():
    args = Namespace()
    print("in pred")
    # args.video = '/home/filipkr/Documents/xjob/vids/real/Videos/MUSSE/musse-SLS/03SLS1R_MUSSE.mts'
    # args.video = '/code/03SLS1R_MUSSE.mts'
    args.out = 'out3'
    print('\n\n\n\n\n\n')
    print(inference_rq.pipe)
    print('\n\n\n\n\n\n')
    job = inference_rq.pipe.queue(args)

    return f"Prediction started!\nTask ({job.id}) added to queue at {job.enqueued_at}"

if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run()
