import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

R_img_file_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\GAN project\\raw_anime"
F_img_file_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\GAN project\\Fake_images"

def create_training_data():
    training_data = []
    for img in os.listdir(R_img_file_path):
            img_array = cv2.imread(os.path.join(R_img_file_path,img))
            img_array = cv2.resize(img_array,(256,256))
            training_data.append([img_array,1])

    for img in os.listdir(F_img_file_path):
            img_array = cv2.imread(os.path.join(F_img_file_path,img))
            img_array = cv2.resize(img_array,(256,256))
            training_data.append([img_array,0])

    return training_data

Train_data= create_training_data()

Train,Test = train_test_split(Train_data, test_size = 0.2, random_state = 42, stratify = Train_data[1])

X_train = []
Y_train = []

X_test = []
Y_test = []

for img,labels in Train:
    X_train.append(img)
    Y_train.append(labels)

for img,labels in Test:
    X_test.append(img)
    Y_test.append(labels)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

#Disciminator
Discriminator = tf.keras.Sequential(name = 'Discriminator')
Discriminator.add(tf.keras.layers.InputLayer(input_shape=(256,256,3)))
Discriminator.add(tf.keras.layers.Conv2D(32, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.keras.activations.relu, padding = 'same'))

# Discriminator.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.MaxPooling2D(2,2))
# Discriminator.add(tf.keras.layers.BatchNormalization())

# Discriminator.add(tf.keras.layers.Conv2D(128, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.Conv2D(128, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.MaxPooling2D(2,2))
# Discriminator.add(tf.keras.layers.BatchNormalization())

# Discriminator.add(tf.keras.layers.Conv2D(256, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.Conv2D(256, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.MaxPooling2D(2,2))
# Discriminator.add(tf.keras.layers.BatchNormalization())

# Discriminator.add(tf.keras.layers.Conv2D(512, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.Conv2D(512, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
# Discriminator.add(tf.keras.layers.MaxPooling2D(2,2))
# Discriminator.add(tf.keras.layers.BatchNormalization())

Discriminator.add(tf.keras.layers.Flatten())

Discriminator.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu))
# Discriminator.add(tf.keras.layers.Dense(128, activation = tf.keras.activations.relu))
Discriminator.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

Discriminator.compile(optimizer = tf.keras.optimizers.Adamax(learning_rate = 0.001), loss = 'binary_crossentropy',metrics=['accuracy', tf.keras.metrics.TruePositives(name='true_positives'),tf.keras.metrics.TrueNegatives(name='true_negatives'),tf.keras.metrics.FalsePositives(name='false_positives'),tf.keras.metrics.FalseNegatives(name='false_negatives')])

Discriminator.summary()

Discriminator.fit(X_train,Y_train)

Discriminator.save("Discriminator.h5")

Discrim = tf.keras.models.load_model("Discriminator.h5")

Discrim.evaluate(X_test,Y_test)