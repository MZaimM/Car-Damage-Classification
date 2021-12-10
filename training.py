import tensorflow as tf
import numpy as np
import os
from prepocessing import Prepocessing

class Training:

    @staticmethod
    def layersCNN():

        layers = [ 
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3),  activation='relu', input_shape=(96,96,1)),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),  activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),  activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=3, activation='softmax')
        ]
        
        return layers


    @staticmethod
    def trainingCNN(train_images, train_labels):
        layers = Training.layersCNN()
        model = tf.keras.Sequential(layers)
        model.compile(optimizer=tf.optimizers.Adam(),
                    loss=tf.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()])
        model.fit(train_images, train_labels, epochs=25, steps_per_epoch=32, verbose = 1)
        model.save_weights("modelCarsDamage.tf")
    
    



# p = Prepocessing()
# train_path = "train/"
# image_files = os.listdir(train_path)
# train_images = [p.load_image(train_path + file) for file in image_files]
# train_labels = [p.extract_label(file) for file in image_files]
# # print(image_files)

# for i in range(len(train_images)):
#     train_images[i] = p.preprocess_image(train_images[i])

# train_images = np.expand_dims(train_images, axis=-1)
# train_labels = np.array(train_labels)
# print(train_images.shape, train_labels.shape)

# # t = Training()

# Training.trainingCNN(train_images=train_images, train_labels=train_labels)
