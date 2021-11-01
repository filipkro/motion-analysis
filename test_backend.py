import requests
from argparse import ArgumentParser, ArgumentTypeError
import time


def create_user(url_base, id):
    error_msgs = [b'User already exists',
                  b'Something went wrong when saving to S3, plz view logs']
    url = url_base + 'create_user'
    data = {'id': id, 'leg': 'R', 'weight': 75, 'length': 185}
    response = requests.post(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin create_user')
            return False
        print(f'create_user response:\n{response.content}')
        print('\ncreate_user ok\n')
        return True
    else:
        print('Error in create_user')
        return False


def get_user(url_base, id):
    error_msgs = [b'No id provided', b'User not in database',
                  b'File could not be downloaded from S3']
    url = url_base + 'get_user'
    data = {'id': id}
    response = requests.get(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin get_user')
            return False
        print(f'get_user response:\n{response.content}')
        print('\nget_user ok\n')
        return True
    else:
        print('Error in get_user')
        return False


def upload_video(url_base, id):
    error_msgs = [b'No id provided', b'No leg provided',
                  b'User not in database', b'Could not save meta data to S3',
                  b'No file part', b'No video selected for uploading',
                  b'File could not be uploaded to S3']
    url = url_base + 'upload'
    file_name = '/home/filipkr/Documents/xjob/vids/real/Videos/' + \
        'MUSSE/musse-SLS/03SLS1R_MUSSE.mts'
    leg = 'R' if 'R' in file_name.split('/')[-1] else 'L'
    data = {'id': id, 'frames': [1, 4, 20, 600], 'leg': leg}
    file = {'file': open(file_name, 'rb')}
    response = requests.post(url, data=data, files=file)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin upload')
            return False
        print(f'upload response:\n{response.content}')
        print('\nupload ok\n')
        return True
    print('Error in upload')
    return False


def assessment_complete(url_base, id):
    error_msgs = [b'No id provided', b'User not in database']
    url = url_base + 'ongoing'
    data = {'id': id}
    total_time = 0
    max_time = 20
    while total_time < max_time:
        print('waiting 60 sec...')
        time.sleep(60)
        response = requests.get(url, data=data)
        if response.ok:
            if response.content in error_msgs:
                print(f'Error: \n{response.content} \nin upload')
                return False
            if response.content == b'Finished':
                print(f'Assessment completed in {total_time} minutes')
                print('\nongoing ok\n')
                return True
        total_time += 1
    print(f'Assessment not completed in {max_time} minutes')
    return False


def get_video(url_base, id):
    error_msgs = [b'No id provided', b'No attempt provided',
                  b'User not in database', b'Attempt not in database',
                  b'File could not be downloaded from S3']
    url = url_base + 'get_video'
    data = {'id': id, 'attempt': 1}
    response = requests.get(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin get_video')
            return False
        if 'vid' in response.headers['Content-Disposition']:
            resp = response.headers['Content-Disposition']
            print(f'Video: {resp}')
            print('\nget_video ok\n')
            return True
    print('Error in get_video')
    return False


def get_result(url_base, id):
    error_msgs = [b'No id provided', b'User not in database',
                  b'Assessment not finished', b'Attempt not in database',
                  b'File could not be downloaded from S3']
    url = url_base + 'get_latest'
    data = {'id': id, 'attempt': 1}
    response = requests.get(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin get_latest')
            return False
        print(f'Latest result: \n{response.content}')
        print('\nget_latest ok\n')
        return True
    print('Error in get_latest')
    return False


def get_all(url_base, id):
    error_msgs = [b'No id provided', b'User not in database',
                  b'Assessment not finished', b'Attempt not in database',
                  b'File could not be downloaded from S3']
    url = url_base + 'get_all'
    data = {'id': id}
    response = requests.get(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin get_latest')
            return False
        print(f'All results: \n{response.content}')
        print('\nget_all ok\n')
        return True
    print('Error in get_all')
    return False


def get_repetition_result(url_base, id):
    error_msgs = [b'No id provided', b'No attempt provided',
                  b'User not in database',
                  b'Assessment not finished', b'Attempt not in database',
                  b'File could not be downloaded from S3']
    url = url_base + 'get_repetition_result'
    data = {'id': id, 'attempt': 1}
    response = requests.get(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin get_repetition_result')
            return False
        print(f'All repetition results: \n{response.content}')
        print('\nget_repetition_result ok\n')
        return True
    print('Error in get_repetition_result')
    return False


def delete_user(url_base, id):
    error_msgs = [b'No id provided', b'Delete unsuccessful']
    url = url_base + 'delete_user'
    data = {'id': id}
    response = requests.post(url, data=data)
    if response.ok:
        if response.content in error_msgs:
            print(f'Error: \n{response.content} \nin delete_user')
            return False
        print(f'Delete user: \n{response.content}')
        print('\ndelete_user ok\n')
        return True
    print('Error in delete_user')
    return False


def main(args):
    if args.heroku:
        url_base = 'https://poe-analysis.herokuapp.com/'
    else:
        url_base = 'http://0.0.0.0:5000/'

    print(f'starting on {url_base}...')
    id = '580612-XXXX'
    # id = '950203'
    completed = 0
    status = True
    status = create_user(url_base, id)
    completed = completed + 1 if status else completed
    if status:
        status = get_user(url_base, id)
        completed = completed + 1 if status else completed
        status = upload_video(url_base, id)
        completed = completed + 1 if status else completed
        if status:
            status = assessment_complete(url_base, id)
            completed = completed + 1 if status else completed
            status = get_video(url_base, id)
            completed = completed + 1 if status else completed
            status = get_result(url_base, id)
            completed = completed + 1 if status else completed
            status = get_all(url_base, id)
            completed = completed + 1 if status else completed
            status = get_repetition_result(url_base, id)
            completed = completed + 1 if status else completed
        status = delete_user(url_base, id)
        completed = completed + 1 if status else completed

    print(f'\n{completed} out of 9 completed successfully')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--heroku', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    main(args)
