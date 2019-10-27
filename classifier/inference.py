import os
import shutil
import pandas as pd
from shot_inference import ShotInference

model_path = 'saved_model/shot_classifier'
image_path = 'test/LCW/'
image_names = os.listdir(image_path)
label_path = 'label/LCW/output_detection_info.csv'

output_path = 'output/'
players = ['LCW', 'TTY']
shots = ['smash', 'drop', 'defense', 'backhand']

label = pd.read_csv(label_path, header=None)

inf_model = ShotInference(model_path)

for img in image_names:
    full_image_path = os.path.join(image_path,img)
    row = label.loc[label[0] == img]
    bbox = (row[2].values[0], row[3].values[0], row[4].values[0], row[5].values[0])
    
    shot = inf_model.predict(0, full_image_path, bbox)[0]
    shot_name = shots[shot]

    out_path = os.path.join(output_path, players[0], shot_name)
    shutil.copy(full_image_path, out_path)