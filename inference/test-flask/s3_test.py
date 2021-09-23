import boto3
from botocore.exceptions import NoCredentialsError
import os
from configparser import ConfigParser

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
BASE = os.path.abspath(os.path.join(BASE, os.pardir))

config = ConfigParser()

config.read(os.path.join(BASE, 'configs/config'))
ACCESS_KEY = config.get('aws', 'access')
SECRET_KEY = config.get('aws', 'secret')


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
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def check_folders(bucket):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    result = s3.list_objects(Bucket=bucket, Prefix='users/filipkro')
    print('Contents' in result)


check_folders('poe-uploads')
# uploaded = upload_to_aws('/home/filipkr/Documents/xjob/vids/real/Videos/Hip-pain/hipp-SLS/02SLS1L.mp4', 'poe-uploads', 'test/02SLS1L.mp4')
# #
# print(uploaded)
# downloaded = download_from_aws('02SLS1L.mp4', 'poe-uploads', 'users/02SLS1L.mp4')
# print(downloaded)
