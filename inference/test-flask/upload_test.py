from flask import Flask, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np


app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_video():
    print('line 43')
    print(request.files)
    print(request.files['file'])
    if 'file' not in request.files:
        print('line 45')
        print(request.url)
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        print('line 49')
        return "No video selected for uploading"
    else:
        print(f'line 52, {file.filename}')
        filename = secure_filename(file.filename)
        print(filename)
        # filename = 'lol.' + filename.split('.')[-1]
        print(file)
        fp = os.path.join('/home/filipkr/', filename)
        file.save(fp)

        cap = cv2.VideoCapture(fp)
        print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
        print('Frame rate: {} fps'.format(fps))


        print('upload_video filename: ' + filename)
        # print(glob.glob(BASE + '/*'))
        # print(os.stat(filename).st_size)




        return "LOL"
