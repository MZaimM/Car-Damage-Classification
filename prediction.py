import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from prepocessing import Prepocessing
import os

from training import Training



class Prediction:

    @staticmethod
    def predict(sample_path, samples):
        t = Training()
        p = Prepocessing()
        eval_images = [p.preprocess_image(p.load_image(sample_path + file)) for file in samples]
        eval_model = tf.keras.Sequential(t.layersCNN())
        eval_model.load_weights("modelCarsDamage.tf")
        eval_predictions = eval_model.predict(np.expand_dims(eval_images, axis=-1))

        cols = 4
        rows = int(np.ceil(len(eval_images)/cols))
        fig = plt.gcf()
        fig.set_size_inches(cols * 4, rows * 4)
        for i in range(len(eval_images)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(eval_images[i], cmap="gray")
            if np.argmax(eval_predictions[i])==2:
                plt.title("Minor")
            elif np.argmax(eval_predictions[i])==1:
                plt.title("Severe")
            else :
                plt.title("Undamage")
            # plt.axis('off')
        plt.show()


sample_path = "sample/"
list = os.listdir(sample_path)
Prediction.predict(sample_path, list)
