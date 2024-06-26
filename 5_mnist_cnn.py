
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

(trainX,trainY),(testX,testY)=tf.keras.datasets.mnist.load_data()

x= np.concatenate((trainX,testX))
y = np.concatenate((trainY,testY))
train_size = 0.75
trainX,testX,trainY,testY= train_test_split(x,y, train_size=train_size)

# Plot a specific MNIST image
def plot_mnist_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('on')
    plt.show()

# Choose the index of the image you want to plot
image_index = 10

# Plot the image
plot_mnist_image(trainX[image_index])

trainX[1].shape

print(trainY[1])

trainY = tf.keras.utils.to_categorical(trainY, num_classes = 10)
testY = tf.keras.utils.to_categorical(testY, num_classes = 10)

print(trainY[1])

model = tf.keras.models.Sequential()

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model.add(Conv2D(64,kernel_size=3, padding="same", activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D())

model.add(Conv2D(32,kernel_size=3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(16,kernel_size=3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(10,activation="softmax"))

model.summary()
#Total Parameters=3×3×1×64+64=576+64=640
#Total Parameters=3×3×64×32+32=18432+32=18464
#Total Parameters=144×10+10=1440+10=1450

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics='accuracy')

model.fit(trainX,trainY,validation_data=(testX,testY),epochs=3)

model.predict(testX[:2])

testY[:2]

plt.imshow(testX[0])

