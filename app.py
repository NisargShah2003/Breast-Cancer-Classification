from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the model
model = load_model("Brest CNN.h5")

def process_image(img):
    img_size = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
    img_array = np.array(img_size)
    img_array = img_array.reshape(-1, 50, 50, 3)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image from the form
        img = request.files['image']
        img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        img_processed = process_image(img)

        # Make the prediction
        prediction = model.predict(img_processed)
        pred_class = np.argmax(prediction, axis=1)

        return render_template('index.html', prediction=pred_class[0])

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
