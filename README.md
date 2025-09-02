# Setup Instructions for multi_modal_emotion_recognition

These steps will help you set up and run the project on macOS, including creating a compatible Python environment for TensorFlow 2.10 and Keras 2.10.

---

## 1. Prerequisites
- Python 3.10.x (installed via asdf, pyenv, or from python.org)
- [asdf](https://asdf-vm.com/) or [pyenv](https://github.com/pyenv/pyenv) (optional, for managing Python versions)

---

## 2. Set Python Version (if using asdf)
```sh
asdf local python 3.10.4
```

---

## 3. Create and Activate Virtual Environment
```sh
python3.10 -m venv tf2env
source tf2env/bin/activate
```

---

## 4. Upgrade pip, setuptools, and wheel
```sh
pip install --upgrade pip setuptools wheel
```

---

## 5. Install TensorFlow and Keras (Apple Silicon/M1/M2)
```sh
pip install tensorflow-macos==2.10.0 keras==2.10.0
```

If you are on Intel Mac, try:
```sh
pip install tensorflow==2.10.0 keras==2.10.0
```

---

## 6. Install Other Dependencies
```sh
pip install opencv-python matplotlib scikit-image pillow pandas
```

---

## 7. Downgrade NumPy if Needed
If you see errors about NumPy 2.x incompatibility, run:
```sh
pip install 'numpy<2'
```

---

## 8. Run the Flask App
```sh
python flask_app.py
```

---

## Troubleshooting
- If you get errors about missing modules, ensure your virtual environment is activated.
- If you get errors about incompatible versions, check your Python version and installed package versions.
- For Apple Silicon, always use `tensorflow-macos`.

---

## Notes
- If you need to re-install Python 3.10.x, use asdf or download from python.org.
- If you need to re-create the environment, delete the `tf2env` folder and repeat the steps above.
