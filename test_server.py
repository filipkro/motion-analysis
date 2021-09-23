from flask import Flask, request
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


@app.route('/get_latest', methods=['GET'])
def get_latest():
    id = request.form['id']

    if not check_user_exist(id):
        return "User does not exist in database"

    attempt = get_attempt_nbr(id) - 1

    if check_ongoing(id, attempt):
        return "Assessment not finished"
    if not check_result_available(id, attempt):
        return "No assessment for user"

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
        for poe in results:
            pred[poe] = results[poe]['pred']
            conf[poe] = tuple(results[poe]['conf'])

        vid = f'users/{id}/ATTEMPT{attempt}/vid.mts'
        utc = get_time_modified(vid)

        res = {'recorded': str(utc), 'pred': pred, 'conf': conf}

        return json.dumps(res)

        # return [f'Result for video uploaded {utc}', json.dumps(res)]

    return "Something went wrong when trying to fetch result from S3"


@app.route('/upload', methods=['POST'])
def upload_video():

    id = request.form['id']
    leg = request.form['leg']

    if not check_user_exist(id):
        return "User not added to database, aborting..."

    attempt = get_attempt_nbr(id)

    frame_splits = request.form.getlist('frames')
    print(id)

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
    params = {'id': id, 'leg': leg, 'weight': weight,
                        'length': length}

    f = open('user_params', 'w')
    json.dump(params, f)
    f.close()

    s3_path = f'users/{id}/user_params.json'
    uploaded = upload_to_aws('user_params', 'poe-uploads', s3_path)

    if uploaded:
        return "User successfully created"
    else:
        return "Something went wrong when saving to S3, plz view logs"


def get_attempt_nbr(id):
    nbr = 1
    base = f'users/{id}/ATTEMPT'

    while file_on_aws(base + str(nbr)):
        nbr += 1

    return nbr


def file_on_aws(file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    return 'Contents' in s3.list_objects(Bucket='poe-uploads',
                                         Prefix=file)


def check_user_exist(id):
    return file_on_aws(f'users/{id}')


def check_ongoing(id, attempt):
    return file_on_aws(f'users/{id}/ATTEMPT{attempt}/ONGOING')


def check_result_available(id, attempt):
    return file_on_aws(f'users/{id}/ATTEMPT{attempt}/results.pkl')


def get_time_modified(file):
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

    print(check_user_exist('950203/vid.mts'))
