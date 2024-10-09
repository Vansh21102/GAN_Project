import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

Latent_dim = 100
n_epochs = 100
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randint(0,255,(latent_dim * n_samples))
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def Train_GAN(n_epochs = n_epochs):
    for i in range(n_epochs):
        GAN_loss = GAN.train_on_batch(generate_latent_points(Latent_dim, 100), np.ones((100,1)))
        print(f'Epoch: {i+1}, GAN Loss: {GAN_loss}')

        if (i+1) % 10 == 0:
            Generator.save(f'generator_epoch_{i+1}.h5')

#Generator
Generator = tf.keras.Sequential(name = 'Generator')
Generator.add(tf.keras.layers.InputLayer(input_shape = Latent_dim))
Generator.add(tf.keras.layers.Dense(128*64*64, activation = tf.keras.activations.relu))
Generator.add(tf.keras.layers.Reshape((64,64,128)))

#Upscaling to 128x128
Generator.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides = (2,2), padding = "same", activation = tf.keras.activations.relu))

#Upscaling to 256x256
Generator.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides = (2,2), padding = "same", activation = tf.keras.activations.relu))

Generator.add(tf.keras.layers.Conv2D(3, (3,3), activation=None, padding='same'))

Generator.summary()

Discriminator = tf.keras.models.load_model('Discriminator.h5')
Discriminator.trainable = False
#GAN
GAN = tf.keras.Sequential(name = 'GAN')
GAN.add(tf.keras.layers.InputLayer(input_shape = Latent_dim))
GAN.add(Generator)
GAN.add(Discriminator)


GAN.compile(optimizer = tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics = ['Accuracy'])

GAN.summary()

Train_GAN(n_epochs = 100)