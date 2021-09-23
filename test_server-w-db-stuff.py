from flask import Flask, request
import redis
from rq import Queue
import sys
import jobs
import os
from werkzeug.utils import secure_filename
# import glob
import psycopg2

DATABASE_URL = os.getenv('DATABASE_URL', 'http://0.0.0.0:5432/')

# db_conn = psycopg2.connect(DATABASE_URL, dbname='videos', password='lol')#sslmode='require')
# db_conn = psycopg2.connect(dbname='videos', password='lol')
# db_conn = psycopg2.connect(host='0.0.0.0', dbname='videos', password='lol')
db_conn = psycopg2.connect(host='db', dbname='videos', user="postgres", password="lol")
db_cursor = db_conn.cursor()
print('PostgreSQL database version:')
db_cursor.execute('SELECT version()')
db_version = db_cursor.fetchone()
print(db_version)
create_command = """CREATE TABLE IF NOT EXISTS videos(
                        video_id SERIAL PRIMARY KEY,
                        video bytea)"""

db_cursor.execute(create_command)

db_cursor.close()
db_conn.close()
del db_cursor
del db_conn
# db_cursor.execute("CREATE TABLE IF NOT EXISTS videos(video_id SERIAL PRIMARY KEY, video bytea)")

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


# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'file' not in request.files:
#         print(request.url)
#         return "No file part"
#     file = request.files['file']
#     if file.filename == '':
#         return "No video selected for uploading"
#     else:
#         filename = secure_filename(file.filename)
#         print(filename)
#         # filename = 'lol.' + filename.split('.')[-1]
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         print('upload_video filename: ' + filename)
#         # print(glob.glob(BASE + '/*'))
#         # print(os.stat(filename).st_size)
#         return f"{filename} uploaded, {predict(vid=filename)}"

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        print(request.url)
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No video selected for uploading"
    else:
        db_conn = psycopg2.connect(host='db', dbname='videos', user="postgres", password="lol")
        db_cursor = db_conn.cursor()
        filename = secure_filename(file.filename)
        print(filename)
        # filename = 'lol.' + filename.split('.')[-1]

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        f2 = open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
        print('upload_video filename: ' + filename)
        # print(glob.glob(BASE + '/*'))
        # print(os.stat(filename).st_size)
        # try:
        f3 = open('/app/03SLS1R_MUSSE.mts', 'rb')
        binary = psycopg2.Binary(f3.read())
        print(f'SIZE OF BINARY:: {sys.getsizeof(f3)}')
        print(f'SIZE OF BINARY:: {sys.getsizeof(f3.read())}')
        print(db_cursor.execute("SELECT pg_size_pretty( pg_database_size('videos') );"))
        insert_command = """INSERT INTO videos(video)
                            VALUES(%s) RETURNING video_id"""
        db_cursor.execute(insert_command, (binary,))
        # db_cursor.execute("INSERT INTO videos(video_id,video) " +
        #         "VALUES(%s,%s)",
        #         (1, binary))
        video_id = db_cursor.fetchone()[0]
        print(f'DATABASE VIDEO_ID: {video_id}')
        db_conn.commit()
        print(db_cursor.execute("SELECT pg_size_pretty( pg_database_size('videos') );"))

        db_cursor.close()
        db_conn.close()
        del db_cursor
        del db_conn
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
