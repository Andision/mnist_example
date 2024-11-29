# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import keras
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image

# model_path = '/data/mnist_saved_model/mnist.keras'

# with strategy.scope():
#     replicated_model = keras.models.load_model(model_path)
#     replicated_model.compile(
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         optimizer=keras.optimizers.Adam(),
#         metrics=['accuracy'])

#     predictions = replicated_model.predict(img_prediction_dataset)
#     scores = tf.nn.softmax(predictions)
#     for path, score in zip(file_paths, scores):
#         print(
#             "The image {} is the number {} with a {:.2f} percent confidence."
#             .format(path, np.argmax(score), 100 * np.max(score))
#         )

# Load the pre-trained MNIST model
model_path = '/data/mnist_saved_model/mnist.keras'
with strategy.scope():
    mnist_model = keras.models.load_model(model_path)
    mnist_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Load the image from the request
        file = request.files['image']
        image = Image.open(BytesIO(file.read()))

        # Preprocess the image
        input_image = preprocess_image(image)

        # Perform prediction
        with strategy.scope():
            predictions = mnist_model.predict(input_image)
        scores = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(scores)
        confidence = 100 * np.max(scores)

        # Return the prediction as JSON
        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)