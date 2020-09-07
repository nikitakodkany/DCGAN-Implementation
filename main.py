#Libraries
import os
import keras
import tfutils
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2DTranspose, Reshape, LeakyReLU
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization

# print("**************")
# print(tf.version)
# print("**************")

# #Image Loading and PreProcessing
# (xtrain, ytrain), (xtest, ytest) = tfutils.datasets.mnist.load_data(one_hot = False)
# xtrain = tfutils.datasets.mnist.load_subset([0], xtrain, ytrain)
# xtest = tfutils.datasets.mnist.load_subset([0], xtest, ytest)
# x = np.concatenate([xtrain, xtest], axis = 0)
# tfutils.datasets.mnist.plot_ten_random_examples(plt, x, np.zeros((x.shape[0], 1))).show()

discriminator = Sequential(
    [
        Conv2D(64, 3, strides=2, input_shape = (28,28,1)), LeakyReLU(), BatchNormalization(),
        Conv2D(128, 5, strides=2), LeakyReLU(), BatchNormalization(),
        Conv2D(256, 3, strides=2), LeakyReLU(), BatchNormalization(),
        Flatten(),
        Dense(1, activation = 'sigmoid')
    ]
)
Optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1 = 0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=['accuracy'])
#discriminator.summary()

"""
Original paper - Input dimention vector: 128.
Input dimention in this network is 1 for simplicity.
"""
generator = Sequential(
    [
        Dense(256, activation='relu', input_shape=(1,)), Reshape((1,1,256)),
        Conv2DTranspose(256, 5, activation='relu'), BatchNormalization(),
        Conv2DTranspose(128, 5, activation='relu'), BatchNormalization(),
        Conv2DTranspose(64, 5, strides=2, activation='relu'), BatchNormalization(),
        Conv2DTranspose(32, 5, activation='relu'), BatchNormalization(),
        Conv2DTranspose(1, 4, activation='sigmoid')
    ]
)
#generator.summary()

# #To check if the generator is working
# noise = np.random.randn(1,1)
# generated_image = generator.predict(noise)[0]
# plt.figure()
# """
# To reshape the generated image as pyplot doesn't accept channels dimention
# as it is a black and white image.
# """
# plt.imshow(np.reshape(generated_image,(28,28)), cmap='binary')
# plt.show()

"""
Deep Convolutional Generative Adversal Network
"""
#noise input
input_layer = tf.keras.layers.Input(shape=(1,))
