import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from object_detection  import  *
#from object_detection import *
#from tensorflow.keras.models import Sequential, load_model

app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()

UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

STATIC_FOLDER = os.path.join(path, 'static')
if not os.path.isdir(STATIC_FOLDER):
    os.mkdir(STATIC_FOLDER)

FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'frames')
if not os.path.isdir(FRAMES_FOLDER):
    os.mkdir(FRAMES_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['avi', 'mp4', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

video = UPLOAD_FOLDER
folder = FRAMES_FOLDER
item = FRAMES_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    print("In here")
    return render_template('upload.html')


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():

    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files[]')

    for file in files:
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            video_to_images(os.path.join(UPLOAD_FOLDER, filename))

    flash('File(s) successfully uploaded')

    #video_to_images(video)
    #search(list, item)
    #detect_objects(folder)

    return redirect('/')

@app.route('/search_objects', methods=["POST", "GET"])
def search_objects():
    search_text = request.form.get("search")
    print(f"Search Text: {search_text}")

    images = search_object(search_text)

    print(f"->{images}")

    return render_template('search.html', images=images)



  

    return "Done"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
