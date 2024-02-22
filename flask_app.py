from flask import Flask,render_template,request,Response
import os
import numpy as np
from numpy import array
import cv2
import pickle
from matplotlib.image import imread
import skimage
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import time
import pandas as pd
app=Flask(__name__)

BASEPATH=os.getcwd()
UPLOAD_PATH=os.path.join(BASEPATH,"static/uploads/")
# loading models
model1 = load_model('/Users/paras/Documents/sem8/btp/web_app/model1.h5')
model2 = load_model('/Users/paras/Documents/sem8/btp/web_app/model2.h5')
model3 = load_model('/Users/paras/Documents/sem8/btp/web_app/model3.h5')

camera=cv2.VideoCapture(0)

# actual_frame_rate = camera.get(cv2.CAP_PROP_FPS)
# print(f"Actual Frame Rate: {actual_frame_rate} fps")


def predict_emotion(frame):
        sample_img=frame

        sample_img=rgb2gray(sample_img)
        sample_img=skimage.transform.resize(sample_img,(197,197))
        sample_img = convert(sample_img)
        sample_img=np.asarray(sample_img)
        if(sample_img[0][0][0]>=1):
            sample_img = sample_img / 255.0
        final=[]
        final.append(sample_img)
        final = np.asarray(final)

        results=[]
        results=model1.predict(final)*0.457+model2.predict(final)*0.134+model3.predict(final)*0.40900000000000003
    
        print("RESULTS",results)
        pred_emotion_idx=np.argmax(results[0])
        if(pred_emotion_idx==0):
            pred_emotion='angry'
        elif(pred_emotion_idx==1):
            pred_emotion='disgust'
        elif(pred_emotion_idx==2):
            pred_emotion='fear'
        elif(pred_emotion_idx==3):
            pred_emotion='happy'
        elif(pred_emotion_idx==4):
            pred_emotion='neutral'
        elif(pred_emotion_idx==5):
            pred_emotion='sad'
        elif(pred_emotion_idx==6):
            pred_emotion='surprise'
        return results,pred_emotion

def generate_frames():
    while True:
        success,frame=camera.read()
        if not success :
            break
        else:
            detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = frame[y:y+h, x:x+w]
                results,v=predict_emotion(face_roi)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, v, (x+6, h ), font, 1.0, (255, 255, 255), 1)
            ret,buffer=cv2.imencode('.jpg',frame)
            face=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + face + b'\r\n')



@app.route('/',methods=['GET','POST'])

def index():
    if(request.method=="POST"):
        upload_file=request.files['my_image']
        filename=upload_file.filename
        print("Uploaded File is ",filename)
        ext=filename.split('.')[-1]
        print("The extension of filename is ",ext)
      
        if(ext.lower() in ['png','jpeg','jpg']):
            
            #send to ML model
            path_save=os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print(upload_file)
            sample_img=imread(upload_file)
            results,v=predict_emotion(sample_img)
            print(results)
            results[0]=np.round(results[0],4)
            top_dict = dict()
            top_dict['angry']=results[0][0]
            top_dict['disgust']=results[0][1]
            top_dict['fear']=results[0][2]
            top_dict['happy']=results[0][3]
            top_dict['neutral']=results[0][4]
            top_dict['sad']=results[0][5]
            top_dict['surprise']=results[0][6]
            
            return render_template('upload.html',fileupload=True,data=top_dict,image_filename=filename)
        else:
            print(ext," file extension not allowed")

    return render_template('upload.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

def convert(img):
    new_img=[]
    for i in range(len(img)):
        t=[]
        for j in range(len(img[i])):
            t.append([img[i][j],img[i][j],img[i][j]])
        new_img.append(t)
    return new_img
def rgb2gray(rgb):
    try:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    except:
        return rgb
if __name__=='__main__':
    app.run(debug=True)