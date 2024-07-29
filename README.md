# Emotion Detection using CNN (Convolutional Neural Network)

## Project Overview

This project leverages deep learning to detect emotions from images using a Convolutional Neural Network (CNN). The final model, ResNet50V2, is trained on the FER-2013 dataset and can recognize seven distinct emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Dataset

The training dataset, [FER-2013](https://www.kaggle.com/datasets/deadskull7/fer2013), is publicly available on Kaggle. It contains labeled facial images representing various emotions.

## Final Model

The chosen architecture for this project is ResNet50V2, an enhanced version of the ResNet family. It uses skip connections to facilitate training and improve model accuracy.

## Applications

This project offers two main applications for emotion detection:

### 1. Gradio App (`app_gradio.py`)
Link: https://huggingface.co/spaces/Npps/Emotion-Detector

This application allows users to upload an image to determine the emotion displayed. It features an intuitive interface powered by Gradio, designed for ease of use by individuals without technical expertise.

To run the Gradio app for emotion detection:

    python app_gradio.py


### 2. OpenCV App (emotion_detect_video.py)
This application uses OpenCV (CV2) to detect emotions in real-time from a live camera feed. It can also analyze emotions from recorded video files, making it versatile for different use cases.

To run the OpenCV app for real-time emotion detection:

    python emotion_detect_video.py

The application will activate the default camera. Press q to exit the application.

## License
This project is licensed under the MIT License.
