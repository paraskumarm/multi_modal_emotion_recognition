
"""Model loading and emotion prediction utilities."""

import numpy as np
import skimage
from tensorflow.keras.models import load_model

# Load models (compile=False for compatibility)
model1 = load_model('models/model1.h5', compile=False)
model2 = load_model('models/model2.h5', compile=False)
model3 = load_model('models/model3.h5', compile=False)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(frame, convert_fn, rgb2gray_fn):
    """Predict emotion from a given image frame."""
    sample_img = rgb2gray_fn(frame)
    sample_img = skimage.transform.resize(sample_img, (197, 197))
    sample_img = convert_fn(sample_img)
    sample_img = np.asarray(sample_img)
    if sample_img[0][0][0] >= 1:
        sample_img = sample_img / 255.0
    final = np.asarray([sample_img])
    # Weighted ensemble prediction
    results = (
        model1.predict(final) * 0.457
        + model2.predict(final) * 0.134
        + model3.predict(final) * 0.409
    )
    pred_emotion_idx = np.argmax(results[0])
    pred_emotion = EMOTIONS[pred_emotion_idx] if pred_emotion_idx < len(EMOTIONS) else 'unknown'
    return results, pred_emotion
