# Skin-Melanoma-Detection-Web-Application
A skin melanoma detection web app developed using Python's Flask framework.

This project implements a web application for detecting skin melanoma using ML. The application allows users to upload images of skin lesions and receive predictions about whether the lesion is indicative of melanoma.

# Setup Instructions
1. `Install Dependencies:` Make sure you have Python installed on your system. You'll also need to install Flask and TensorFlow. You can install them using pip.
2. `Run the Application:` Execute the app.py file to start the Flask web server.

# Usage
- `Home Page:` The home page (`/`) provides a simple interface for users to navigate to other sections of the application.
- `About Page:`The about page (`/about`) provides information about the project and its objectives.
- `Skin Melanoma Detection:` Users can access the skin melanoma detection feature by visiting the prediction page (`/efficientnet`). They can upload an image of a skin lesion and receive predictions about whether the lesion is indicative of melanoma.
- `Prediction Result:` After uploading an image, users are redirected to the prediction result page (`/prediction_result`), where they can view the prediction outcome.

# Project Structure
- `app.py:` The main Flask application file containing routes and logic for the web application.
- `templates/:` This directory contains HTML templates for different pages of the web application.
- `models/:` Trained deep learning `EfficientNetB0` model for skin melanoma detection.
- `labels.txt:` Text file containing labels for prediction classes.
