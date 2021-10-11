from flask import Flask, request, send_file
import redis
from rq import Queue
# import sys
import jobs
import os
import json
from werkzeug.utils import secure_filename
import boto3
from configparser import ConfigParser
from botocore.exceptions import NoCredentialsError
from dateutil import tz
from numpy import argmax
import re
import datetime

import inference_rq as inference_rq # noqa

if os.environ.get("REDIS_URL") is not None:
    r = redis.from_url(os.environ.get("REDIS_URL"))
else:
    r = redis.from_url('redis://redis:6379')

# print('BEFORE APP \n\n\n\n\n LOL')
app = Flask(__name__)
q = Queue(connection=r)

BASE = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
config = ConfigParser()

config.read(os.path.join(BASE, 'configs/config'))
ACCESS_KEY = config.get('aws', 'access')
SECRET_KEY = config.get('aws', 'secret')

del config, ConfigParser

# VIDEOS = tuple("mp4 mov mts avi")
# videos = UploadSet('videos', VIDEOS)
app.config['UPLOAD_FOLDER'] = '/data'
app.config['UPLOAD_EXTENSIONS'] = ['mts', 'mp4', 'mpeg4', 'avi']
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.secret_key = "secret key"


@app.route('/get_video', methods=['GET'])
def get_video():
    id = request.form.get('id')
    attempt = request.form.get('attempt')
    # print(request.)
    if id is None:
        print('args')
        id = request.args.get('id')
    if id is None:
        print('values')
        id = request.values.get('id')
    if id is None:
        return "No id provided"
    if attempt is None:
        print('args')
        attempt = request.args.get('attempt')
    if attempt is None:
        print('values')
        attempt = request.values.get('attempt')
    if attempt is None:
        return "No attempt provided"

    # id = '950203'
    # attempt = 1

    if not check_user_exist(id):
        return "User not in database"

    prefix = f'users/{id}/ATTEMPT{attempt}/vid'

    if not file_on_aws(prefix):
        return "Attempt not in database"

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    s3_file = s3.list_objects(Bucket='poe-uploads',
                              Prefix=prefix)['Contents'][0]['Key']

    file = 'vid.' + s3_file.split('.')[-1]

    downloaded = download_from_aws(file, 'poe-uploads', s3_file)

    if downloaded:
        return send_file(file, download_name='vid.' + s3_file.split('.')[-1],
                         mimetype='video/MP2T')
        # return "file sent ?"

    return 'File could not be downloaded from S3'


@app.route('/get_user', methods=['POST', 'GET'])
def get_user():
    id = request.form.get('id')
    # print(request.)
    if id is None:
        print('args')
        id = request.args.get('id')
    if id is None:
        print('values')
        id = request.values.get('id')
    if id is None:
        return "No id provided"
    print(request.form)
    print(request.args)
    print(request.args.get('id'))
    print(request.values)
    print(request.values.get('id'))
    # attempt = request.form['attempt']
    # id = '950203'
    # attempt = 1

    print(f'get user with ID: {id}')

    if not check_user_exist(id):
        print('user not in database')
        return "User not in database"

    s3_file = f'users/{id}/user_params.json'

    file = 'user_params.json'

    downloaded = download_from_aws(file, 'poe-uploads', s3_file)

    if downloaded:
        f = open(file, 'r')
        data = json.load(f)
        f.close()
        return str(data)
        # return "file sent ?"

    return 'File could not be downloaded from S3'


@app.route('/get_latest', methods=['POST', 'GET'])
def get_latest():
    id = request.form.get('id')
    # print(request.)
    if id is None:
        print('args')
        id = request.args.get('id')
    if id is None:
        print('values')
        id = request.values.get('id')
    if id is None:
        return "No id provided"

    if not check_user_exist(id):
        return "User not in database"

    attempt = get_attempt_nbr(id) - 1
    if check_ongoing(id, attempt):
        return "Assessment not finished"
    if not check_result_available(id, attempt):
        return "No assessment for this attempt"

    res = get_results(id, attempt)
    if type(res) == dict:
        return json.dumps(res)
    else:
        return res


@app.route('/get_all', methods=['GET'])
def get_all_results():
    id = request.form.get('id')
    # print(request.)
    if id is None:
        print('args')
        id = request.args.get('id')
    if id is None:
        print('values')
        id = request.values.get('id')
    if id is None:
        return "No id provided"

    if not check_user_exist(id):
        return "User not in database"

    last_attempt = get_attempt_nbr(id) - 1
    results = {}

    for attempt in range(1, last_attempt):
        if check_ongoing(id, attempt) or \
                not check_result_available(id, attempt):

            results[f'attempt{attempt}'] = 'Not available'
        else:
            results[f'attempt{attempt}'] = get_results(id, attempt)

    return json.dumps(results)


@app.route('/get_repetition_result', methods=['GET'])
def get_repetition():
    id = request.form.get('id')
    attempt = request.form.get('attempt')
    # print(request.)
    if id is None:
        print('args')
        id = request.args.get('id')
    if id is None:
        print('values')
        id = request.values.get('id')
    if id is None:
        return "No id provided"
    if attempt is None:
        print('args')
        attempt = request.args.get('attempt')
    if attempt is None:
        print('values')
        attempt = request.values.get('attempt')
    if attempt is None:
        return "No attempt provided"

    if not check_user_exist(id):
        return "User not in database"
    if check_ongoing(id, attempt):
        return "Assessment not finished"
    if not check_result_available(id, attempt):
        return "No assessment for this attempt"

    res = get_results(id, attempt, with_reps=True)
    if type(res) == dict:
        return json.dumps(res)
    else:
        return res


def get_results(id, attempt, with_reps=False):

    file = 'result.pkl'
    downloaded = download_from_aws(file, 'poe-uploads',
                                   f'users/{id}/ATTEMPT{attempt}/results.pkl')

    if downloaded:
        import pickle

        f = open(file, 'rb')
        results = pickle.load(f)
        f.close()

        pred = {}
        conf = {}
        reps = {}
        for poe in results:
            pred[poe] = results[poe]['pred']
            conf[poe] = tuple(results[poe]['conf'])
            if with_reps:
                individual = results[poe]['detailed']
                reps[poe] = {}
                for i, rep in enumerate(individual):
                    reps[poe][f'rep{i}'] = int(argmax(rep))
                    reps[poe][f'rep{i}confs'] = tuple(rep)

        vid = f'users/{id}/ATTEMPT{attempt}/vid.mts'
        utc = get_modified_time(vid)

        if with_reps:
            return {'time': str(utc), 'pred': pred, 'conf': conf, 'reps': reps}
        else:
            return {'time': str(utc), 'pred': pred, 'conf': conf}

    return "Something went wrong when trying to fetch result from S3"


@app.route('/upload', methods=['POST'])
def upload_video():

    id = request.form.get('id')
    leg = request.form.get('leg')
    # print(request.)
    if id is None:
        print('args')
        id = request.args.get('id')
    if id is None:
        print('values')
        id = request.values.get('id')
    if id is None:
        return "No id provided"
    if leg is None:
        print('args')
        leg = request.args.get('leg')
    if leg is None:
        print('values')
        leg = request.values.get('leg')
    if leg is None:
        return "No leg provided"

    if not check_user_exist(id):
        return "User not in database"

    attempt = get_attempt_nbr(id)

    frame_splits = request.form.getlist('frames')
    print(id)

    meta = {'leg': leg, 'frames': tuple(frame_splits)}

    meta_name = 'meta'
    f = open(meta_name, 'w')
    json.dump(meta, f)
    f.close()

    s3_path = f'users/{id}/ATTEMPT{attempt}/meta.json'
    uploaded = upload_to_aws(meta_name, 'poe-uploads', s3_path)

    if not uploaded:
        return "Could not save meta data to S3"

    # assert False
    if 'file' not in request.files:
        print(request.url)
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No video selected for uploading"
    else:
        filename = secure_filename(file.filename)
        print(filename)

        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join('/app', filename)
        file.save(file_path)
        file.close()

        new_name = f'users/{id}/ATTEMPT{attempt}/vid.' + filename.split('.')[-1]
        uploaded = upload_to_aws(file_path, 'poe-uploads', new_name)
        os.remove(file_path)
        if uploaded:
            print('upload_video filename: ' + filename)
            return f"{filename} uploaded,\n{predict(vid=new_name, id=id, leg=leg, attempt=attempt)}"
        else:
            print("not uploaded to aws correctly")
            return "not uploaded to aws correctly"


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


@app.route("/create_user", methods=["POST"])
def create_user():
    id = request.form['id']
    leg = request.form['leg']
    weight = request.form['weight']
    length = request.form['length']

    if check_user_exist(id):
        # overwrite previous information??
        return "User already exists"

    # config = ConfigParser()
    # config['user'] = request.form
    params = {'id': id, 'leg': leg, 'weight': weight, 'length': length}

    f = open('user_params', 'w')
    json.dump(params, f)
    f.close()

    s3_path = f'users/{id}/user_params.json'
    uploaded = upload_to_aws('user_params', 'poe-uploads', s3_path)

    if uploaded:
        return "User successfully created"
    else:
        return "Something went wrong when saving to S3, plz view logs"


def fix_id(id):
    id = re.sub('[^0-9]', '', str(id))
    current = datetime.date.today().year
    if not (id[0] == '1' or id[0] == '2'):
        if len(id) == 6 or len(id) == 10:
            year = int(id[:2])
            id = '20' + id if year < current % 100 else '19' + id
        else:
            return id, 'Wrong id'
    if len(id) == 8:
        id = id + 'XXXX'
        return id, 'Wrong id, add four digits'

    return id, ''


def get_attempt_nbr(id):
    nbr = 1
    base = f'users/{id}/ATTEMPT'

    while file_on_aws(base + str(nbr)):
        nbr += 1

    return nbr


def file_on_aws(file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    return 'Contents' in s3.list_objects(Bucket='poe-uploads', Prefix=file)


def check_user_exist(id):
    return file_on_aws(f'users/{id}')


def check_ongoing(id, attempt):
    return file_on_aws(f'users/{id}/ATTEMPT{attempt}/ONGOING')


def check_result_available(id, attempt):
    return file_on_aws(f'users/{id}/ATTEMPT{attempt}/results.pkl')


def get_modified_time(file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    to_zone = tz.gettz('Europe/Stockholm')
    utc = s3.list_objects(Bucket='poe-uploads',
                          Prefix=file)['Contents'][0]['LastModified']
    return utc.astimezone(to_zone)


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def download_from_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.download_file(bucket, s3_file, local_file)
        del s3
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def get_result_for_user(id):
    return 0


def predict(vid='03SLS1R_MUSSE.mts', id='', leg='R', attempt=1):
    if id != '':
        job = q.enqueue(inference_rq.pipe, args=(vid, id, leg, attempt),
                        job_timeout=-1)

        return (f"Prediction for {vid} started!\nTask ({job.id})" +
                " added to queue at {job.enqueued_at}")
    else:
        return "no id"


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    # app.run()
    # get_results('950203', 1, with_reps=True)

    print(check_user_exist('950203/ATTEMPT1/vid'))
