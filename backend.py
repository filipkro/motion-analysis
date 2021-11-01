from flask import Flask, request, send_file
import os
import json
from werkzeug.utils import secure_filename
import boto3
from configparser import ConfigParser
import backend_utils

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
    """
        Download video, specified by id and attempt.
        ---
        parameters:
          - in: form-data
            schema:
              type: object
              properties:
                id:
                  type: string
                  description: unique identification number
                attempt:
                  type: string
                  description: specifies which attempt to get, if not specified - latest video is downloaded
        responses:
          - 200:
              descprition: Successfully downloaded file
              content:
                type: binary
                description: video file
            400:
              description: No id provided
              content:
                type: string
                description: error message
            404:
              description: Requested user or attempt not found
              content:
                type: string
                description: error message
            501:
              description: Error when downloading files from S3
              content:
                type: string
                description: error message
            500:
              content:
                type: string
                description: error message
    """
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400
    attempt = backend_utils.get_variable_from_req(request, 'attempt')
    if attempt is None:
        attempt = backend_utils.get_attempt_nbr(id) - 1

    if not backend_utils.check_user_exist(id):
        return "User not in database", 404

    prefix = f'users/{id}/ATTEMPT{attempt}/vid'

    if not backend_utils.file_on_aws(prefix):
        return "Attempt not in database", 404

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    s3_file = s3.list_objects(Bucket='poe-uploads',
                              Prefix=prefix)['Contents'][0]['Key']

    file = 'vid.' + s3_file.split('.')[-1]

    downloaded, error = backend_utils.download_from_aws(file, s3_file)

    if downloaded:
        return send_file(file, download_name='vid.' + s3_file.split('.')[-1],
                         mimetype='video/MP2T'), 200
        # return "file sent ?"

    return error, 501


@app.route('/delete_user/<id>', methods=['DELETE'])
def delete_user(id=None):
    """
        Delete user, including all data, from database, specified by id.
        ---
        parameters:
          - id: unique identification number
        responses:
          - 200:
              descprition: Successfully downloaded file
              content: [binary] video file
          - 400:
              description: No id provided
              content: [string] error message
          - 404:
              description: Requested user or attempt not found
              content: [string] error message
          - 501:
              description: Error when downloading files from S3
              content: [string] error message
          - 500:
              description: Error
    """
    # id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400

    print(f'DELETE {id}')

    if not backend_utils.check_user_exist(id):
        print('user not in database')
        return "User not in database", 404

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    files = s3.list_objects(Bucket='poe-uploads',
                            Prefix=f'users/{id}')['Contents']
    keys = [{'Key': key['Key']} for key in files]
    keys.append({'Key': f'users/{id}'})
    keys = {'Objects': keys}

    deleted = s3.delete_objects(Bucket='poe-uploads', Delete=keys)

    if deleted:
        print(f'Successfully deleted user {id}')
        return 'Delete successful', 200
    else:
        print(f'Could not delete user {id}')
        return 'Delete unsuccessful', 501


@app.route('/get_user', methods=['GET'])
def get_user():
    '''
        Download user information, such as injured leg, length, weight.
        Specified by id.
    '''
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400

    print(f'get user with ID: {id}')

    if not backend_utils.check_user_exist(id):
        print('user not in database')
        return "User not in database", 404

    s3_file = f'users/{id}/user_params.json'

    file = 'user_params.json'

    downloaded, error = backend_utils.download_from_aws(file, s3_file)

    if downloaded:
        f = open(file, 'r')
        data = json.load(f)
        f.close()
        return str(data), 200
        # return "file sent ?"

    return error, 501


@app.route('/get_result', methods=['GET'])
def get_result():
    '''
        Download latest result, specified by id.
    '''

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400

    if not backend_utils.check_user_exist(id):
        return "User not in database", 404

    attempt = backend_utils.get_variable_from_req(request, 'attempt')
    if attempt is None:
        attempt = backend_utils.get_attempt_nbr(id) - 1

    if backend_utils.check_ongoing(id, attempt):
        return "Assessment not finished", 201
    if not backend_utils.check_result_available(id, attempt):
        return "Attempt not in database", 404

    res = backend_utils.get_results(id, attempt)
    if type(res) == dict:
        return json.dumps(res), 200
    else:
        return res, 501


@app.route('/ongoing', methods=['GET'])
def get_ongoing():
    '''
        Check if assessment is ongoing for user, specified by id.
        Returns 'Ongoing' if assessment is ongoing and 'Finished' if it is
        completed.
    '''
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400

    if not backend_utils.check_user_exist(id):
        return "User not in database", 404

    attempt = backend_utils.get_attempt_nbr(id) - 1
    if backend_utils.check_ongoing(id, attempt):
        return "Ongoing", 200
    else:
        return "Finished", 201


@app.route('/get_all', methods=['GET'])
def get_all_results():
    '''
        Download all results for user, specified by id.
    '''

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400

    if not backend_utils.check_user_exist(id):
        return "User not in database", 404

    last_attempt = backend_utils.get_attempt_nbr(id)
    results = []

    for attempt in range(1, last_attempt):
        if backend_utils.check_ongoing(id, attempt) or \
                not backend_utils.check_result_available(id, attempt):

            #     results[f'attempt{attempt}'] = 'Not available'
            # else:
            #     results[f'attempt{attempt}'] = backend_utils.get_results(id,
            #                                                              attempt)
            results.append('Not available')
        else:
            results.append(backend_utils.get_results(id, attempt))

    return json.dumps(results), 200


@app.route('/get_repetition_result', methods=['GET'])
def get_repetition():
    '''
        Download predictions for all repetitions, specified by id and attempt.
    '''

    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400
    attempt = backend_utils.get_variable_from_req(request, 'attempt')
    if attempt is None:
        attempt = backend_utils.get_attempt_nbr(id) - 1

    if not backend_utils.check_user_exist(id):
        return "User not in database", 404
    if backend_utils.check_ongoing(id, attempt):
        return "Assessment not finished", 201
    if not backend_utils.check_result_available(id, attempt):
        return "Attempt not in database", 404

    res = backend_utils.get_results(id, attempt, with_reps=True)
    if type(res) == dict:
        return json.dumps(res), 200
    else:
        return res, 500


@app.route('/upload', methods=['POST'])
def upload_video():
    '''
        Upload video to assess. Specify id and leg
        (and frames between repetitions, TO BE IMPLEMENTED).
        Video provided as file in files.
    '''
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        print("No id provided")
        return "No id provided", 400
    leg = backend_utils.get_variable_from_req(request, 'leg')
    if leg is None:
        print("No leg provided")
        return "No leg provided", 400

    if not backend_utils.check_user_exist(id):
        print("User not in database")
        return "User not in database", 404

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
        print("Could not save meta data to S3")
        return "Could not save meta data to S3", 501

    if 'file' not in request.files:
        print(request.url)
        print("no file in files")
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        print("no filename")
        return "No video selected for uploading", 400
    else:
        filename = secure_filename(file.filename)
        print(filename)

        file_path = os.path.join('/app', filename)
        file.save(file_path)
        file.close()

        s3_name = f'users/{id}/ATTEMPT{attempt}/vid.' + filename.split('.')[-1]
        uploaded = backend_utils.upload_to_aws(file_path, s3_name)
        os.remove(file_path)
        if uploaded:
            print('upload_video filename: ' + filename)
            status = backend_utils.predict(s3_name, id, leg, attempt=attempt)
            return f"{filename} uploaded,\n{status}", 200
        else:
            print("not uploaded to aws correctly")
            return "File could not be uploaded to S3", 501


@app.route("/")
def hello_world():
    return "<p>Hello, World!\nLOL1</p>"


@app.route("/create_user", methods=["POST"])
def create_user():
    '''
        Create user in database, provide id, injured leg, weight, and length.
    '''
    id = backend_utils.get_variable_from_req(request, 'id')
    if id is None:
        return "No id provided", 400
    leg = request.form.get('leg')

    weight = request.form.get('weight')
    length = request.form.get('length')

    if backend_utils.check_user_exist(id):
        # overwrite previous information??
        return "User already exists", 401

    params = {'id': id, 'leg': leg, 'weight': weight, 'length': length}

    f = open('user_params', 'w')
    json.dump(params, f)
    f.close()

    s3_path = f'users/{id}/user_params.json'
    uploaded = backend_utils.upload_to_aws('user_params', s3_path)

    if uploaded:
        return "User successfully created", 200
    else:
        return "Something went wrong when saving to S3, plz view logs", 501


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    # app.run()
    # get_results('950203', 1, with_reps=True)

    print(backend_utils.check_user_exist('950203/ATTEMPT1/vid'))
