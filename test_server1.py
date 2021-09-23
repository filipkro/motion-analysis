from flask import Flask, request
import redis
from rq import Queue
# import sys
import jobs
import os
from werkzeug.utils import secure_filename
import glob

import inference_rq as inference_rq # noqa

if os.environ.get("REDIS_URL") is not None:
    r = redis.from_url(os.environ.get("REDIS_URL"))
else:
    r = redis.from_url('redis://redis:6379')

# print('BEFORE APP \n\n\n\n\n LOL')
app = Flask(__name__)
q = Queue(connection=r)

BASE = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# VIDEOS = tuple("mp4 mov mts avi")
# videos = UploadSet('videos', VIDEOS)
app.config['UPLOAD_FOLDER'] = '/data'
app.config['UPLOAD_EXTENSIONS'] = ['mts', 'mp4', 'mpeg4', 'avi']
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.secret_key = "secret key"
# print('AFTER APP \n\n\n\n\n LOL')
# inference_rq.rq.init_app(app)
# print('BEFORE INIT_APP \n\n\n\n\n LOL')

# jobs.rq.init_app(app)


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        print(request.url)
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No video selected for uploading"
    else:
        filename = secure_filename(file.filename)
        print(filename)
        # filename = 'lol.' + filename.split('.')[-1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_video filename: ' + filename)
        # print(glob.glob(BASE + '/*'))
        # print(os.stat(filename).st_size)
        return f"{filename} uploaded, {predict(vid=filename)}"


@app.route("/")
def hello_world():
    return "<p>Hello, World!\nLOL1</p>"


@app.route("/task")
def add_task():

    if request.args.get("n"):

        # job = jobs.background_task.queue(request.args.get("n"))
        job = q.enqueue(jobs.background_task, request.args.get("n"))

        return f"Task ({job.id}) added to queue at {job.enqueued_at}"

    return "No value for count provided"


@app.route("/predict")
def predict_entry():
    print("in pred")
    # args.video = '/home/filipkr/Documents/xjob/vids/real/Videos/MUSSE/musse-SLS/03SLS1R_MUSSE.mts'
    # args.video = '/code/03SLS1R_MUSSE.mts'
    # out = 'out3'
    # print('\n\n\n\n\n\n')
    # print(inference_rq.pipe)
    # print('\n\n\n\n\n\n')
    # job = inference_rq.pipe.queue(args)

    return predict()


def predict(vid='03SLS1R_MUSSE.mts', out='out'):
    job = q.enqueue(inference_rq.pipe, args=(out, vid,), job_timeout=-1)

    return f"Prediction for {vid} started!\nTask ({job.id}) added to queue at {job.enqueued_at}"

if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run()
