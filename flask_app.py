
"""Flask app for multi-modal emotion recognition."""

import os
import numpy as np
from flask import Flask, render_template, request, Response
from matplotlib.image import imread
from utils import convert, rgb2gray
from video_stream import generate_frames
from models import predict_emotion


app = Flask(__name__)

# Paths
BASEPATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASEPATH, "static/uploads/")

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page for file upload and emotion prediction."""
    if request.method == "POST":
        upload_file = request.files['my_image']
        filename = upload_file.filename
        ext = filename.split('.')[-1]
        if ext.lower() in ['png', 'jpeg', 'jpg']:
            # Save uploaded file
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            sample_img = imread(upload_file)
            results, _ = predict_emotion(sample_img, convert, rgb2gray)
            results[0] = np.round(results[0], 4)
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            top_dict = {emotions[i]: results[0][i] for i in range(len(emotions))}
            return render_template(
                'upload.html', fileupload=True, data=top_dict, image_filename=filename
            )
        print(ext, " file extension not allowed")
        return render_template('upload.html')
    return render_template('upload.html')

@app.route('/video')
def video():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)
