import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import random
import tensorflow as tf


# Function to load images and labels
def load_data(directory, labels, exclude_folder):
    images = []
    image_labels = []

    # Iterate over the folders in the directory
    for label, num_label in labels.items():
        folder_path = os.path.join(directory, label)

        if os.path.basename(folder_path) == exclude_folder:
            # Skip the excluded folder
            continue

        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png')):
                image_path = os.path.join(folder_path, image_file)
                image = load_img(image_path, target_size=(224, 224))
                image_array = img_to_array(image)
                image_array /= 255.0  # Normalize the image
                images.append(image_array)
                image_labels.append(num_label)

    return np.array(images), np.array(image_labels)


# Function to apply random flips and noise to the image
def preprocess_image(image):
    # Randomly apply vertical and horizontal flips
    if random.random() < 0.5:
        image = tf.image.flip_left_right(image)
    if random.random() < 0.5:
        image = tf.image.flip_up_down(image)
    # Add random noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image += noise
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


# Function to make the agent's decision
def make_decision(model, img):
    noisy_img = preprocess_image(img)
    prediction = model.predict(np.expand_dims(noisy_img, axis=0))
    predicted_class = np.argmax(prediction)

    # Define a mapping from the predicted class to the winning move
    winning_moves = {0: 2, 1: 0, 2: 1}  # rock(0) -> paper(2), scissors(1) -> rock(0), paper(2) -> scissors(1)
    winning_move = winning_moves[predicted_class]
    return winning_move


# Function to simulate the agent's decision in one round
def play_one_round(model, X_test, y_test):
    # Select a random image and preprocess it
    random_idx = np.random.choice(len(X_test))
    image = X_test[random_idx]
    true_label = y_test[random_idx]

    # Predict the move
    predicted_move = make_decision(model, image)

    # Define winning relationships (rock: 0, scissors: 1, paper: 2)
    winning_moves = {0: 1, 1: 2, 2: 0}


    # Determine the outcome and calculate profit
    if predicted_move == true_label:
        return 0  # Draw
    elif winning_moves[predicted_move] == true_label:
        return 1  # Win
    else:
        return -1  # Lose


# Function to play multiple rounds
def play_game(rounds, model, X_test, y_test):
    profits = []
    for _ in range(rounds):
        profit = play_one_round(model, X_test, y_test)
        profits.append(profit if profits == [] else profits[-1] + profit)
    return profits

