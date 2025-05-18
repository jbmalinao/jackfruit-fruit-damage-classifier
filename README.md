## Overview
The Jackfruit Fruit Damage Classifier is a web application built with Streamlit that allows users to upload images of jackfruits and classify them into one of four categories: "Fruit_borer," "Fruit_fly," "Healthy," or "Rhizopus_rot." The application uses a pre-trained Convolutional Neural Network (CNN) for feature extraction, followed by a Support Vector Machine (SVM) for classification.

## Features
*   **Image Upload:** Users can easily drag and drop or browse to upload JPG, JPEG, or PNG images of jackfruits.
*   **Damage Classification:** Predicts the type of damage or if the fruit is healthy.
*   **Confidence Score:** Displays the confidence level of the prediction if the SVM model supports probability estimates.
*   **Detailed Probabilities:** Provides a bar chart of probabilities for each class (if available).
*   **User-Friendly Interface:** Simple and intuitive UI powered by Streamlit.
*   **Responsive Design:** Adapts to different screen sizes for usability on desktop and mobile.

## Technology Stack
*   **Frontend:** Streamlit
*   **Backend & Core Logic:** Python
*   **Machine Learning:**
    *   TensorFlow (Keras API) for the CNN feature extractor.
    *   Scikit-learn for the SVM classifier and feature scaler.
*   **Image Processing:** Pillow (PIL), NumPy
*   **Model & Scaler Persistence:** Joblib (`.pkl`), HDF5 (`.h5`)
*   **File Handling:** Pathlib
*   **Styling:** Custom CSS

## How It Works
The classification process involves several steps:
1.  **Image Input:** The user uploads an image.
2.  **Preprocessing:**
    *   The image is resized to `(224, 224) pixels`.
    *   It's converted to a NumPy array.
    *   Grayscale images are converted to 3-channel RGB.
    *   Alpha channels (if present) are removed.
    *   The array is preprocessed using `mobilenet_v2_preprocess_input` (which typically scales pixel values to the range [-1, 1]).
3.  **Feature Extraction:** The preprocessed image is fed into a pre-trained CNN (MobileNetV2-based) which acts as a feature extractor, outputting a feature vector.
4.  **Feature Scaling:** The extracted features are scaled using a pre-trained `StandardScaler` to normalize them.
5.  **Classification:** The scaled features are passed to a pre-trained SVM model, which predicts the class label.
6.  **Output:** The predicted class label, along with confidence and detailed probabilities is displayed to the user.

## Setup and Installation
To run this project locally, follow these steps:

1.  **Prerequisites:**
    *   Python 3.8 - 3.11 (TensorFlow 2.13.0, which was in your logs, officially supports up to Python 3.11. If using newer TensorFlow, Python 3.12 might be fine).
    *   `pip` (Python package installer)

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd jackfruit-fruit-damage-classifier
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Ensure your `requirements.txt` file contains the necessary packages:
    ```txt
    streamlit
    Pillow
    numpy
    tensorflow-cpu  # Or tensorflow if you have a GPU and CUDA configured
    scikit-learn
    # joblib is usually included with scikit-learn
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ensure Model Files are Present:**
    Make sure `jackfruit_cnn_feature_extractor.h5`, `jackfruit_feature_scaler.pkl`, and `jackfruit_svm_classifier.pkl` are in the root directory of the project or update the paths in `app.py` accordingly.

6.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    The application should open in your default web browser.

## Usage Guide
1.  **Open the Application:** Access the URL provided when you run `streamlit run app.py` (usually `http://localhost:8501`).
2.  **Upload an Image:**
    *   Drag and drop a jackfruit image (JPG, JPEG, PNG) onto the designated area.
    *   Or, click "Browse files" to select an image from your computer.
3.  **View Selected Image:** The uploaded image will be displayed in the right column.
4.  **Predict:** Click the "Predict" button.
5.  **View Results:**
    *   The application will display the "Diagnosis" (predicted condition).
    *   A "Confidence" score for the prediction will be shown.
    *   An expandable section "View Detailed Probabilities" will show a bar chart of probabilities for all classes, if the SVM model was trained with `probability=True`.

