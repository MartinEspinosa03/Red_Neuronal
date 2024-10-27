import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__) 
CORS(app)

model_path = os.path.join('models', 'modelo.h5')

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"No file or directory found at {model_path}")

model = load_model(model_path)
classes = ['Cadena', 'Calcetas', 'Camisa', 'Camiseta',
    'Corbata', 'Gorra', 'Lentes_sol',
    'Manga_Larga', 'Pans', 'Pantalon', 'Playera', 'Pulsera', 'Tenis']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':  
    app.run(debug=True, port=5000)