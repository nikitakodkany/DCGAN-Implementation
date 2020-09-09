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

#Image Loading and PreProcessing
print("")
print('Dataset loading...')
(xtrain, ytrain), (xtest, ytest) = tfutils.datasets.mnist.load_data(one_hot = False)
print('Dataset loaded successfully!')
print("")
xtrain = tfutils.datasets.mnist.load_subset([0], xtrain, ytrain)
xtest = tfutils.datasets.mnist.load_subset([0], xtest, ytest)
x = np.concatenate([xtrain, xtest], axis = 0)
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
# print("")
# print('Discriminator Summary')
#discriminator.summary()

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
# print("")
# print('Generator Summary')
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

#noise input
input_layer = tf.keras.layers.Input(shape=(1,))
generator_out = generator(input_layer)
discriminator_out = discriminator(generator_out)
gan = Model(input_layer, discriminator_out)
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=['accuracy'])
# print("")
# print('GAN SUMMARY')
#gan.summary()

epoch = 25
batch_size = 128
steps_per_epoch = int(2*x.shape[0]/batch_size)
# print('Steps per Epoch = ', steps_per_epoch)

dynamic_plotting = tfutils.plotting.DynamicPlot(plt,5,5,(8,8))

for e in range(0, epoch):
    dynamic_plotting.start_of_epoch(e)
    for step in range(0, steps_per_epoch):
        true_example = x[int(batch_size/2)*step:int(batch_size/2)*(step+1)]
        true_example = np.reshape(true_example,(true_example.shape[0],28,28,1))
        noise = np.random.randn(int(batch_size/2),1)
        generator_example = generator.predict(noise)

        #batches
        xbatch = np.concatenate([generator_example,true_example],axis=0)
        ybatch = np.array([0]*int(batch_size/2)+[1]*int(batch_size/2))

        #randomize the order
        indices = np.random.choice(range(batch_size), batch_size, replace = False)

        xbatch = xbatch[indices]
        ybatch = ybatch[indices]

        discriminator.trainable=True
        discriminator.train_on_batch(xbatch, ybatch)
        discriminator.trainable=False

        loss, _ = gan.train_on_batch(noise, np.ones(int(batch_size/2)))
        _, acc = discriminator.evaluate(xbatch, ybatch, verbose=False)

    noise = np.random.randn(1,1)
    generated_image = generator.predict(noise)[0]
    # dynamic_plotting.end_of_epoch(generated_image,'binary','Discriminator Acc.:{:2f}'.format(acc),'Gan Loss:{:2f}'.format(loss))
