

from flask import Flask, render_template, request, Response
import os
import numpy as np
from matplotlib.image import imread


app = Flask(__name__)

# Paths
BASEPATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASEPATH, "static/uploads/")


# Import model prediction logic
from models import predict_emotion






# Import video streaming logic
from video_stream import generate_frames



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['my_image']
        filename = upload_file.filename
        ext = filename.split('.')[-1]
        if ext.lower() in ['png', 'jpeg', 'jpg']:
            # Save uploaded file
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            sample_img = imread(upload_file)
            results, pred_emotion = predict_emotion(sample_img, convert, rgb2gray)
            results[0] = np.round(results[0], 4)
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            top_dict = {emotions[i]: results[0][i] for i in range(len(emotions))}
            return render_template('upload.html', fileupload=True, data=top_dict, image_filename=filename)
        else:
            print(ext, " file extension not allowed")
    return render_template('upload.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Import utility functions
from utils import convert, rgb2gray
if __name__ == '__main__':
    app.run(debug=True)