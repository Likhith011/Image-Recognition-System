import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
