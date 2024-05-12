# -*- coding: utf-8 -*-
"""deep learning milestone 3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D2K8mD3tAJpFdGMMVCWh8BAgoK9Rgz8C
"""

#name: urba muskaan
#student id:URB22608789

from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory paths for the train and test sets
train_dir = '/content/drive/MyDrive/new dataset chest x ray/chest_xray/train'
test_dir = '/content/drive/MyDrive/new dataset chest x ray/chest_xray/test'

# Create an ImageDataGenerator for preprocessing and loading images
datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]

# Define batch size
batch_size = 32

# Create the train and test generators
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=batch_size,
    class_mode='binary'  # Assuming it's a binary classification task
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary'
)

model_path = model_path = '/content/drive/MyDrive/xray.h5/chest_xray_classifier.h5'

from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model(model_path)

# Verify that the model was loaded successfully
print("Model loaded successfully!")

# Evaluate the loaded model on the test set
evaluation_results = loaded_model.evaluate(test_generator)

# Print the evaluation results
print("Evaluation Results:")
print("Loss:", evaluation_results[0])
print("Accuracy:", evaluation_results[1])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate predictions on the test set
predictions = loaded_model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Get true labels
true_labels = test_generator.classes

# Calculate confusion matrix
cm = confusion_matrix(true_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the classes
classes = ['Class 1', 'Class 2',]  # Define your class names here

# Generate predictions on the test set
predictions = loaded_model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Get true labels
true_labels = test_generator.classes

# Calculate confusion matrix
cm = confusion_matrix(true_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Get the unique class labels and their counts
unique_classes, class_counts = np.unique(train_generator.classes, return_counts=True)

# Plot the class distribution
plt.figure(figsize=(8, 6))
plt.bar(unique_classes, class_counts)
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution in Training Set')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get true labels from the test set
true_labels = test_generator.classes

# Calculate predictions
predictions = loaded_model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

#data analysis
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory paths for the train and test sets
train_dir = '/content/drive/MyDrive/new dataset chest x ray/chest_xray/train'
test_dir = '/content/drive/MyDrive/new dataset chest x ray/chest_xray/test'

# Get the class labels
classes = os.listdir(train_dir)

# Count the number of images in each class
train_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in classes]
test_counts = [len(os.listdir(os.path.join(test_dir, c))) for c in classes]

# Plot the class distribution
plt.figure(figsize=(10, 5))
plt.bar(classes, train_counts, label='Train')
plt.bar(classes, test_counts, label='Test', alpha=0.5)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Training and Test Sets')
plt.legend()
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define train directory
train_dir = '/content/drive/MyDrive/new dataset chest x ray/chest_xray/train'

# Create ImageDataGenerator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    rotation_range=40,  # Randomly rotate images by 40 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20% of the width
    height_shift_range=0.2,  # Randomly shift images vertically by 20% of the height
    shear_range=0.2,  # Shear intensity (shear angle in radians)
    zoom_range=0.2,  # Randomly zoom images by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Strategy for filling in newly created pixels
)

# Define batch size and target image size
batch_size = 32
target_size = (256, 256)

# Generate batches of training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'  # Assuming binary classification
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the input shape based on the target size of your images
input_shape = (256, 256, 3)  # Assuming RGB images

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Define the number of epochs
epochs = 10

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

import matplotlib.pyplot as plt

# Get training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs_range = range(1, epochs + 1)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)

# Print the evaluation results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

from tensorflow.keras.layers import Dropout

def create_model_with_dropout(input_shape, dropout_rate=0.3):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),  # Add dropout layer
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the number of epochs
epochs = 10

# Train the model with dropout layers enabled
dropout_model = create_model_with_dropout(input_shape=train_generator.image_shape)
dropout_history = dropout_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Evaluate the trained model on the test dataset
test_loss_dropout, test_accuracy_dropout = dropout_model.evaluate(test_generator)

# Compare the performance metrics with the previous model without dropout
print("Test Loss (with dropout):", test_loss_dropout)
print("Test Accuracy (with dropout):", test_accuracy_dropout)

# Analyze the outcomes and identify areas for enhancement
# Compare the test loss and accuracy between the model with dropout and the previous model without dropout
# Look for improvements in accuracy and reductions in overfitting

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), dropout_history.history['loss'], label='Training Loss (with dropout)')
plt.plot(range(1, epochs + 1), dropout_history.history['val_loss'], label='Validation Loss (with dropout)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), dropout_history.history['accuracy'], label='Training Accuracy (with dropout)')
plt.plot(range(1, epochs + 1), dropout_history.history['val_accuracy'], label='Validation Accuracy (with dropout)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()