from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers.legacy import Adam  # Use the legacy Adam optimizer
import numpy as np
from PIL import UnidentifiedImageError
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('podcast_image_classifier.h5', custom_objects={'Adam': Adam})

# Class labels
class_labels = {0: '001', 1: '002'}  # Update this if needed

# Product information
product_info = {
    'A0': {
        "ID": "A0",
        "Name": "Soft Smoothing Seamless: Turtleneck Top",
        "Brand": "SKIMS",
        "price": 64.00,
        "Status": "Fast Selling",
        "img": "static/img_1.0.jpg"
    },
    'A1': {
        "ID": "A1",
        "Name": "Slim Fit Long Sleeve Turtleneck Top",
        "Brand": "Norma Kamali",
        "price": 95.00,
        "Status": "Available",
        "img": "static/img_1.1.jpg"
    },
    'A2': {
        "ID": "A2",
        "Name": "Ribbed Sea Coast Mockneck Long Sleeve",
        "Brand": "Alo",
        "price": 68.00,
        "Status": "Available",
        "img": "static/img_1.2.jpg"
    },
    'B0': {
        "ID": "B0",
        "Name": "Le Papier La Chemise Papier Shirt",
        "Brand": "Jacquemus",
        "price": 735.00,
        "Status": "SOLD OUT",
        "img": "static/img_2.0.jpg"
    },
    'B1': {
        "ID": "B1",
        "Name": "Men's Teca Ecru Button Down Shirt",
        "Brand": "Portuguese Flannel",
        "price": 130.00,
        "Status": "Available",
        "img": "static/img_2.1.jpg"
    },
    'B2': {
        "ID": "B2",
        "Name": "Textured Cotton Shirt",
        "Brand": "Zara",
        "price": 69.90,
        "Status": "Available",
        "img": "static/img_2.2.jpg"
    },
    'B3': {
        "ID": "B3",
        "Name": "Cotton Shirt",
        "Brand": "Club Monaco",
        "price": 60.00,
        "Status": "Available",
        "img": "static/img_2.3.jpg"
    }
}

def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        img_file = request.files['image']
        if not img_file:
            return jsonify({'error': 'No image provided'}), 400

        # Read the image file and convert it to a PIL image
        img_bytes = img_file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(150, 150))
        
        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale the image as done in training

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = int(prediction[0][0] > 0.5)  # Assuming binary classification with sigmoid activation
        predicted_label = class_labels[predicted_class]

        # Return product information based on the predicted label
        if predicted_label == '002':
            result = [product_info['A0'], product_info['A1'], product_info['A2']]
        else:
            result = [product_info['B0'], product_info['B1'], product_info['B2'], product_info['B3']]

        for product in result:
            product['img'] = get_image_as_base64(product['img'])

        return jsonify({'predicted_class': predicted_label, 'output': result}), 200
    except UnidentifiedImageError:
        return jsonify({'error': 'Cannot identify image file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change port if 5000 is in use