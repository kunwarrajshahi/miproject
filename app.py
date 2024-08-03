from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from pridictions import getPredictions 

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    
    img_bb, entities = getPredictions(image)
    
    _, img_encoded = cv2.imencode('.png', img_bb)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    response = {
        "entities": entities,
        "image": img_base64
    }
    
    return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

