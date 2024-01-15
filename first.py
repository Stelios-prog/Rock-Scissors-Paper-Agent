import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from my_model import create_model
import matplotlib.pyplot as plt
import random

from functions import load_data, preprocess_image, play_game, make_decision
from my_model import create_model

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Assign numerical values to the moves
labels = {'rock': 0, 'scissors': 1, 'paper': 2}
data_dir = 'archive'
exclude_folder = 'rps-cv-images'

# Load the data
images, image_labels = load_data(data_dir, labels, exclude_folder)

# Split the data into 70% training, 20% validation, and 10% testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    images, image_labels, test_size=0.3, stratify=image_labels, random_state=42)
test_size_proportion = 0.1 / 0.3
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=test_size_proportion, stratify=y_temp, random_state=42)

# Model creation
input_shape = X_train.shape[1:]  # This will be (224, 224, 3)
model = create_model(input_shape)

# Callbacks for training
callbacks = [ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1),
             ReduceLROnPlateau(factor=0.33, patience=2,verbose=1),
             EarlyStopping(patience=5, verbose=1)]

# Data augmentation generator
train_datagen = ImageDataGenerator(
    rotation_range=60,  # degrees
    width_shift_range=0.2,  # fraction of total width
    height_shift_range=0.2,  # fraction of total height
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,  # Number of batches per epoch
    epochs=100,
    callbacks=callbacks,
    validation_data=(X_val, y_val)
)

# Load the best model and evaluate on the test set
model = load_model("best_model.h5")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("test_loss:", test_loss, "test_acc:", test_accuracy)

# Generate predictions for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Play the game for N rounds
N = 100
profits_over_time = play_game(N, model, X_test, y_test)

# Plot the agent's profit over time
plt.plot(profits_over_time)
plt.title('Agent Profit Over Time')
plt.xlabel('Round')
plt.ylabel('Total Profit (€)')
plt.show()


