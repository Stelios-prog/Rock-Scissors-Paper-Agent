# Rock-Paper-Scissors Game Agent - README

## Introduction
This project implements a machine learning model to play the Rock-Paper-Scissors game. It utilizes TensorFlow and Keras to train a deep learning model that recognizes and responds to images representing rock, paper, or scissors gestures.

## Features
- Image preprocessing for better model training.
- Implementation of Convolutional Neural Network (CNN) using Keras.
- Data augmentation to enhance the training dataset.
- Evaluation using confusion matrix and prediction accuracy.
- Game simulation against a randomly acting agent.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Installation
1. Clone the repository: `git clone https://github.com/Stelios-prog/Rock-Scissors-Paper-Agent`.
2. Install required packages: `pip install -r requirements.txt`.

## Usage
1. Load the dataset: The dataset should be in the format as specified in the Kaggle dataset.
2. Train the model: Run the training script to train the model using the dataset.
3. Test the model: Evaluate the model's performance on the test dataset.
4. Play the game: Simulate the game for a specified number of rounds and observe the model's performance.

## Dataset
This project uses a dataset from Kaggle, containing over 700 images each for rock, paper, and scissors gestures.

## Model Architecture
The model architecture includes multiple convolutional layers, max-pooling layers, a flattening layer, dropout for regularization, and dense layers for final classification.

## Training
The training involves data augmentation techniques and uses callbacks like ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping.

## Evaluation
Performance is evaluated using a confusion matrix and accuracy metrics on the test dataset.

## Game Simulation
The game simulation involves the model making predictions against a random agent, with the financial outcome based on wins, draws, and losses.

## Contribution
Feel free to fork the repository and contribute to the project.

## License
This project is licensed under the MIT License.

## Contact
For any queries, please contact papargirisstelios3@gmail.com.
