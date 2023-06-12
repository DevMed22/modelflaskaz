from PIL import Image
import io
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model


app = Flask(__name__)

# Define the predict function
model = load_model('covid_vgg.h5')
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the HTTP request
    image_data = request.files['image'].read()

    # Create PIL image object from bytes data
    img = Image.open(io.BytesIO(image_data))

    # Preprocess the image data as necessary
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32')
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run the image data through your ML model
    predictions = model.predict(img_array)

    # Return the model's predictions as JSON
    return jsonify(predictions.tolist())
@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"
if __name__ == '__main__':
    app.run(debug=True)
