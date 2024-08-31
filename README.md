# Emotional Speech Recognition (Deep Learning)

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

- This project focuses on recognizing emotions from speech using deep learning (LSTM) techniques.
- It utilizes the Toronto Emotional Speech Set (TESS) dataset, which contains recordings of emotional speech in various contexts.
- The model is designed to classify emotions from audio signals, with a focus on providing accurate emotion detection from different speakers and emotional states.
- The project uses TensorFlow and Keras for building and training the deep learning model.

## Data

- **Dataset**: Toronto Emotional Speech Set (TESS)
  - The dataset includes recordings of emotionally expressive speech, with labels indicating different emotions.
  - It is available on Kaggle and has been downloaded locally for this project.
  - **Data Folder Structure**:
    ```
    - dataset/
      - [audio folders with files and annotations]
    ```
  - **Data Preprocessing**:
    - **Feature Extraction**: Mel-frequency cepstral coefficients (MFCCs) are extracted from the audio files to represent the audio signal.
    - **Feature Scaling**: StandardScaler is used to normalize the features for better model performance (Audio frequencies can often vary greatly)

## Features

- **Emotion Classification**: The model classifies emotions from audio recordings into categories such as happy, sad, angry, etc.
- **Model Checkpointing**: The best-performing model based on validation accuracy is saved during training.
- **Early Stopping**: Training is halted if the model's performance does not improve on the validation set for a specified number of epochs.

## Installation

To set up and run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/mohammed-majid/speech-emotion-recognition.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd speech-emotion-recognition
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the dataset**:
    - Ensure the TESS dataset is downloaded and placed in the `dataset/` folder.

5. **Run the training script**:
    ```bash
    python model.ipynb
    ```

## Usage

1. **Feature Extraction**:
    - Extract MFCC features from audio files using the provided preprocessing script.

2. **Training the Model**:
    - The model is trained using the `model.ipynb` script.
    - Checkpoints are saved as `best_model.keras` in the project directory.

3. **Evaluating the Model**:
    - The `model.py` contains a cell that visualizes the learning curves to demonstrate the model's performance on training and testing datasets

4. **Predicting Emotions**:
    - Use the trained model to predict emotions from new audio samples.
    - Ensure the new audio files are preprocessed in the same way as the training data.

## Acknowledgements

This project was developed using the following libraries and tools:
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Librosa](https://librosa.org/)
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)

### Side Note
- The dataset used for this project is available on Kaggle. You can access it [here](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).
