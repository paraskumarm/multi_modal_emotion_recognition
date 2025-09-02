import cv2
from flask import Response
from models import predict_emotion
from utils import convert, rgb2gray

# Initialize camera
camera = cv2.VideoCapture(0)

def generate_frames():
    """Video streaming generator function."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(frame, 1.1, 7)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame[y:y + h, x:x + w]
            _, emotion = predict_emotion(face_roi, convert, rgb2gray)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, emotion, (x + 6, h), font, 1.0, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def release_camera():
    if camera.isOpened():
        camera.release()
