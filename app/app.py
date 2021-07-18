import os
from pathlib import Path
from re import T

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from cv_predictor import breed

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "720ec2f076"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
upload = Path(UPLOAD_FOLDER)
upload.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def identify_page():
    return render_template('identify.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file provided, please select a file')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No file provided, please select a file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        uploadFilename = secure_filename(file.filename)
        print(uploadFilename)
        filepath = os.path.join(UPLOAD_FOLDER, uploadFilename)
        file.save(filepath)
        predicted_breed = breed(filepath)
        flash('Your Prediction:')
        return render_template('identify.html', filename=uploadFilename, prediction=predicted_breed)
    else:
        flash('Image type is not allowed. Please upload an image of type: {0}'.format(
            ALLOWED_EXTENSIONS))
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print(filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
