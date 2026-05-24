from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Check if model exists
model_path = 'model/handwriting_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

# Labels: 0-9 digits mapped to A-J for demo
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Open and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28 (MNIST format)
        img = img.resize((28, 28))
        img_array = np.array(img)
        
        # Normalize to 0-1
        img_array = img_array / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

        prediction = model.predict(img_array, verbose=0)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return jsonify({
            'letter': labels[index],
            'confidence': confidence,
            'all_predictions': {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
        })
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
