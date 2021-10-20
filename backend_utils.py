import os
import boto3
from configparser import ConfigParser
from botocore.exceptions import NoCredentialsError
from dateutil import tz
from numpy import argmax
import re
import datetime
from rq import Queue
import redis

import inference_rq as inference_rq # noqa

if os.environ.get("REDIS_URL") is not None:
    r = redis.from_url(os.environ.get("REDIS_URL"))
else:
    r = redis.from_url('redis://redis:6379')

q = Queue(connection=r)

BASE = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
config = ConfigParser()

config.read(os.path.join(BASE, 'configs/config'))
ACCESS_KEY = config.get('aws', 'access')
SECRET_KEY = config.get('aws', 'secret')

del config, ConfigParser


def get_variable_from_req(request, key):
    var = request.form.get(key)
    if var is None:
        print('args')
        var = request.args.get(key)
    if var is None:
        print('values')
        var = request.values.get(key)

    return var


def get_results(id, attempt, with_reps=False):

    file = 'result.pkl'
    downloaded = download_from_aws(file,
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

    return "File could not be downloaded from S3"


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


def upload_to_aws(local_file, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    bucket = 'poe-uploads'
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


def download_from_aws(local_file, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    bucket = 'poe-uploads'
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

    print(check_user_exist('950203/ATTEMPT1/vid'))
