from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename

from prediction import Prediction
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():   
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No File part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            flash('Format Gambar harus png, jpg, jpeg, gif')
            return redirect(request.url)
    list = os.listdir(UPLOAD_FOLDER)
    Prediction.predict(UPLOAD_FOLDER, file_names)
    return render_template('index.html', filenames = file_names)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename, ), code=301)



 
if __name__ == "__main__":
    app.run()