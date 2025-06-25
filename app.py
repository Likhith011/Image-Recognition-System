from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils import preprocess_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('model/cnn_model.keras')
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image = preprocess_image(filepath)
            pred = model.predict(image)
            prediction = classes[np.argmax(pred)]
            filename = file.filename
    return render_template('index.html', prediction=prediction, filename=filename)


if __name__ == '__main__':
    import webbrowser
    from threading import Timer

    # Launch the browser after a short delay
    Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()

    # Run the Flask app
    app.run(debug=True)

