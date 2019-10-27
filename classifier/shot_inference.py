import os
import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array

class ShotInference:
    def __init__(self, model_path):
        self.inference = load_model(model_path)
        self.image_row = 128
        self.image_col = 128

    def _resize_image(self, image):
        input_img_resized = cv2.resize(image,(self.image_row, self.image_col))
        input_img_resized = input_img_resized.astype('float')/255.0
        input_img_resized = img_to_array(input_img_resized)
        input_img_resized = np.expand_dims(input_img_resized, axis=0)

        return input_img_resized

    def _process_cropped_image(self, image_path):
        input_img = cv2.imread(image_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        return self._resize_image(input_img)

    def _process_non_cropped_image(self, image_path, bbox_coor):
        input_img = cv2.imread(image_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        
        x_min = bbox_coor[0]
        y_min = bbox_coor[1]
        x_max = bbox_coor[2]
        y_max = bbox_coor[3]

        cropped_img = input_img[x_min:x_max, y_min:y_max]
        
        return self._resize_image(cropped_img)
        # then, we need to get only the positions

    def cropped_image(self, image_path, bbox_coor):
        input_img = cv2.imread(image_path)        
        x_min = bbox_coor[0]
        y_min = bbox_coor[1]
        x_max = bbox_coor[2]
        y_max = bbox_coor[3]

        cropped_img = input_img[x_min:x_max, y_min:y_max]
        
        return cropped_img
        # then, we need to get only the positions
        
        
    def predict(self, cropped, image_path, bbox_coor):
        if(cropped):
            # if cropped? only take the image and send it
            img = self._process_cropped_image(image_path)
            return self.inference.predict_classes(img),img
        else:
            # if not cropped? we need to know where the box is
            img = self._process_non_cropped_image(image_path, bbox_coor)
            return self.inference.predict_classes(img),img