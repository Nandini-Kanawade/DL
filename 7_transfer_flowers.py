# -*- coding: utf-8 -*-
"""transfer_flowers

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yF2NjlS0Sy_9-HttBVVW7Hc8tuUujk0m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pathlib

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten,Activation, BatchNormalization,MaxPooling2D
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow_datasets as tfds

(train_ds, train_labels), (test_ds,test_labels) = tfds.load(
    "tf_flowers",
    split =["train[:70%]","train[:30%]"],
    batch_size=-1,
    as_supervised=True,


)

train_ds=tf.image.resize(train_ds,(150,150))

test_ds=tf.image.resize(test_ds,(150,150))

print(train_ds.shape)

print(test_ds.shape)

print(train_labels.shape)

print(train_labels[0]
      )

import tensorflow as tf
import matplotlib.pyplot as plt

# Load the tf_flowers dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Iterate through the dataset to display images
for example in dataset.take(5):  # Display the first 5 images
    image = example["image"]
    label = example["label"]

    # Convert the image tensor to a numpy array
    image = image.numpy()

    # Display the image
    plt.imshow(image)
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
    plt.show()



train_labels=tf.keras.utils.to_categorical(train_labels,num_classes=5)
test_labels=tf.keras.utils.to_categorical(test_labels,num_classes=5)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

base_model=VGG16(weights="imagenet",include_top=False,
                 input_shape = train_ds[0].shape)
base_model.trainable = False

train_ds=preprocess_input(train_ds)

test_ds=preprocess_input(test_ds)

base_model.summary()

from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()

dense_layer_1 = layers.Dense(50, activation='relu')

dense_layer_2 = layers.Dense(20,activation='relu')

prediction_layer=layers.Dense(5,activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds,train_labels,epochs=2,validation_data=(test_ds,test_labels),batch_size=32)

"""Final Accuracy = 99.30%"""

# Assuming model is your trained model
# Example: model = tf.keras.models.Sequential([...])

# Make predictions for a single example from the test dataset
example_index = 1  # Index of the example you want to predict

# Get the input data (image) for the example
input_data = test_ds[example_index][0]  # Assuming test_ds is a tuple (input_data, label)
input_data = input_data[np.newaxis, ...]  # Add batch dimension (model expects batches)

# Make the prediction
predictions = model.predict(input_data)

# Get the predicted class (assuming classification task)
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Print the predicted class index
print("Predicted class index:", predicted_class_index)