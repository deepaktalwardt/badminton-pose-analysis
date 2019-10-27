import os
import shutil
import pandas as pd
from classifier.shot_inference import ShotInference
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, Markdown, Latex,HTML
import matplotlib.pylab as plt
import cv2
print(os.getcwd())
from open_pose_model.pose_predictor import Pose_predictor

pose_model_path = os.path.join(os.getcwd(), 'saved_model/keras_openpose_model.h5')

pose_detector = Pose_predictor(pose_model_path)