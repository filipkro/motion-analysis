from flask import Flask, request, send_file
import os
import json
from werkzeug.utils import secure_filename
import boto3
from configparser import ConfigParser

import backend_utils

import inference_rq # noqa


app = Flask(__name__)

BASE = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
config = ConfigParser()

config.read(os.path.join(BASE, 'configs/config'))
ACCESS_KEY = config.get('aws', 'access')
SECRET_KEY = config.get('aws', 'secret')

del config, ConfigParser

app.config['UPLOAD_FOLDER'] = '/data'
app.config['UPLOAD_EXTENSIONS'] = ['mts', 'mp4', 'mpeg4', 'avi']
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.secret_key = "secret key"


@app.route('/get_video', methods=['GET'])
def get_video():
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided"
    attempt = backend_utils.get_variable_from_req(request, 'attempt')
    if attempt is None:
        return "No attempt provided"

    if not backend_utils.check_user_exist(id):
        return "User not in database"

    prefix = f'users/{id}/ATTEMPT{attempt}/vid'

    if not backend_utils.file_on_aws(prefix):
        return "Attempt not in database"

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    s3_file = s3.list_objects(Bucket='poe-uploads',
                              Prefix=prefix)['Contents'][0]['Key']

    file = 'vid.' + s3_file.split('.')[-1]

    downloaded = backend_utils.download_from_aws(file, s3_file)

    if downloaded:
        return send_file(file, download_name='vid.' + s3_file.split('.')[-1],
                         mimetype='video/MP2T')
        # return "file sent ?"

    return 'File could not be downloaded from S3'


@app.route('/get_user', methods=['POST', 'GET'])
def get_user():
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided"

    print(f'get user with ID: {id}')

    if not backend_utils.check_user_exist(id):
        print('user not in database')
        return "User not in database"

    s3_file = f'users/{id}/user_params.json'

    file = 'user_params.json'

    downloaded = backend_utils.download_from_aws(file, s3_file)

    if downloaded:
        f = open(file, 'r')
        data = json.load(f)
        f.close()
        return str(data)
        # return "file sent ?"

    return 'File could not be downloaded from S3'


@app.route('/get_latest', methods=['POST', 'GET'])
def get_latest():

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided"

    if not backend_utils.check_user_exist(id):
        return "User not in database"

    attempt = backend_utils.get_attempt_nbr(id) - 1
    if backend_utils.check_ongoing(id, attempt):
        return "Assessment not finished"
    if not backend_utils.check_result_available(id, attempt):
        return "No assessment for this attempt"

    res = backend_utils.get_results(id, attempt)
    if type(res) == dict:
        return json.dumps(res)
    else:
        return res


@app.route('/get_all', methods=['GET'])
def get_all_results():

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided"

    if not backend_utils.check_user_exist(id):
        return "User not in database"

    last_attempt = backend_utils.get_attempt_nbr(id) - 1
    results = {}

    for attempt in range(1, last_attempt):
        if backend_utils.check_ongoing(id, attempt) or \
                not backend_utils.check_result_available(id, attempt):

            results[f'attempt{attempt}'] = 'Not available'
        else:
            results[f'attempt{attempt}'] = backend_utils.get_results(id,
                                                                     attempt)

    return json.dumps(results)


@app.route('/get_repetition_result', methods=['GET'])
def get_repetition():

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided"
    attempt = backend_utils.get_variable_from_req(request, 'attempt')
    if attempt is None:
        return "No attempt provided"

    if not backend_utils.check_user_exist(id):
        return "User not in database"
    if backend_utils.check_ongoing(id, attempt):
        return "Assessment not finished"
    if not backend_utils.check_result_available(id, attempt):
        return "No assessment for this attempt"

    res = backend_utils.get_results(id, attempt, with_reps=True)
    if type(res) == dict:
        return json.dumps(res)
    else:
        return res


@app.route('/upload', methods=['POST'])
def upload_video():

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided"
    leg = backend_utils.get_variable_from_req(request, 'leg')
    if leg is None:
        return "No leg provided"

    if not backend_utils.check_user_exist(id):
        return "User not in database"

    attempt = backend_utils.get_attempt_nbr(id)

    frame_splits = request.form.getlist('frames')
    print(id)

    meta = {'leg': leg, 'frames': tuple(frame_splits)}

    meta_name = 'meta'
    f = open(meta_name, 'w')
    json.dump(meta, f)
    f.close()

    s3_path = f'users/{id}/ATTEMPT{attempt}/meta.json'
    uploaded = backend_utils.upload_to_aws(meta_name, s3_path)

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
        uploaded = backend_utils.upload_to_aws(file_path, new_name)
        os.remove(file_path)
        if uploaded:
            print('upload_video filename: ' + filename)
            return f"{filename} uploaded,\n{backend_utils.predict(vid=new_name, id=id, leg=leg, attempt=attempt)}"
        else:
            print("not uploaded to aws correctly")
            return "not uploaded to aws correctly"


@app.route("/")
def hello_world():
    return "<p>Hello, World!\nLOL1</p>"


@app.route("/create_user", methods=["POST"])
def create_user():
    id = request.form['id']
    leg = request.form['leg']
    weight = request.form['weight']
    length = request.form['length']

    if backend_utils.check_user_exist(id):
        # overwrite previous information??
        return "User already exists"

    params = {'id': id, 'leg': leg, 'weight': weight, 'length': length}

    f = open('user_params', 'w')
    json.dump(params, f)
    f.close()

    s3_path = f'users/{id}/user_params.json'
    uploaded = backend_utils.upload_to_aws('user_params', s3_path)

    if uploaded:
        return "User successfully created"
    else:
        return "Something went wrong when saving to S3, plz view logs"


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    # app.run()
    # get_results('950203', 1, with_reps=True)

    print(backend_utils.check_user_exist('950203/ATTEMPT1/vid'))
