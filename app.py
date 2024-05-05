from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
import tempfile

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = './uploads'

def load_models():
    # Specify the path to the models folder
    models_folder = "models"
    
    # Load the model
    efficientnet_model_path = os.path.join(models_folder, "efficientnet_model.h5")
    efficientnet_model = load_model(efficientnet_model_path)
    
    return efficientnet_model

def preprocess_image(file_storage):
    # Save the uploaded file to a temporary location
    temp_file, temp_path = tempfile.mkstemp(suffix=".jpg")
    file_storage.save(temp_path)

    # Load the image and preprocess it
    image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize the image

    # Remove the temporary file
    os.close(temp_file)
    os.remove(temp_path)

    return input_arr

def efficientnet_prediction(efficientnet_model, test_image):
    efficientnet_predictions = efficientnet_model.predict(test_image)
    return np.argmax(efficientnet_predictions)

def read_labels():
    with open("labels.txt") as f:
        content = f.readlines()
    label = [i[:-1] for i in content]
    return label


@app.route('/')
def home():
    # Render the home page
    return render_template('home.html')

@app.route('/about')
def about():

    # Render the about page
    return render_template('about.html')

@app.route('/efficientnet', methods=['GET', 'POST'])
def efficientnet_prediction_route():

    if request.method == 'POST':
        test_image = request.files['test_image']
        efficientnet_model = load_models()
        input_arr = preprocess_image(test_image)
        result_index = efficientnet_prediction(efficientnet_model, input_arr)
        labels = read_labels()
        return render_template('prediction_result.html', result=labels[result_index], image_path=test_image.filename)
    return render_template('efficientnet.html')

@app.route('/prediction_result', methods=['GET', 'POST'])
def prediction_result_route():
        # Redirect to the home page
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
