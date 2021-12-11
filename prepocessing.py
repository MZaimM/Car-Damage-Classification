import os
import cv2
import matplotlib.pyplot as plt
import numpy 

class Prepocessing:
    
    @staticmethod
    def load_image(file_path):
        return cv2.imread(file_path)

    @staticmethod
    def extract_label(file_name):
        if "minor" in file_name:
            return 2;
        elif "severe" in file_name:
            return 1;
        else:
            return 0

    @staticmethod
    def greyScale(img):
        """
        b --> blue
        g --> green
        r --> red 
        """
        b, g, r = cv2.split(img)
        greyImage = 0.299*r + 0.587*g + 0.114*b  #Rumus Greyscale
        src3 =  numpy.round(greyImage).astype('int') 
        return src3

    @staticmethod
    def preprocess_image(img, side=96):
        x, y = img.shape[0], img.shape[1]
        min_side = min(x, y)
        img = img[:min_side, :min_side] #cropping
        img = cv2.resize(img, (side,side)) #resizing

        p = Prepocessing()
        img = p.greyScale(img) #greyscaling
        return img / 255.0

# # Melakukan load dataset
# p = Prepocessing()
# train_path = "train/"
# image_files = os.listdir(train_path)
# train_images = [p.load_image(train_path + file) for file in image_files]
# train_labels = [p.extract_label(file) for file in image_files]

# # Melakukan preview prepocessing
# preview_index = 199
# plt.subplot(1,2,1)
# plt.imshow(train_images[preview_index])
# plt.subplot(1,2,2)
# plt.imshow(p.preprocess_image(train_images[preview_index]), cmap="gray") 
# plt.show()

