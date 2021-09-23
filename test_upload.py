import os
# from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import glob


UPLOAD_FOLDER = './'
BASE = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
print(glob.glob(BASE + '/*'))
if '/app/lol.mp4' in glob.glob(BASE + '/*'):
    print(os.stat('lol.mp4').st_size)

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        print(request.url)
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        print(filename)
        filename = 'lol.' + filename.split('.')[-1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        print(glob.glob(BASE + '/*'))
        print(os.stat(filename).st_size)
        return render_template('upload.html', filename=filename)


@app.route('/display/<filename>')
def display_video(filename):
    print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
