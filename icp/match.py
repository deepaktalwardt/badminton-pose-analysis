import os
import cv2
import json
import pandas as pd
import numpy as np
from numpy.linalg import norm
import icp
import random


data_path = 'data/LCW/backhand/'
best_smash = 'unlabelled_LCW_Malaysia2017_20_result.png'
best_smash_name_only = 'unlabelled_LCW_Malaysia2017_20'

label_path = 'label/LCW/output_detection_info.csv'
label = pd.read_csv(label_path, header=None)

image_names = os.listdir(data_path)

reference = cv2.imread(os.path.join(data_path, best_smash)) # get reference image

# get the correct joint information
data = ''
with open('LCW_backhand.json', "r") as read_file:
    data = json.load(read_file)

POINTS = {}

for key in data:
    # key is the filename
    large_list = data[key]
    
    get_real_person = large_list[0]

    for item in large_list:
        if(len(item) > len(get_real_person)):
            get_real_person = item
    
    # iterate dictionary
    point = {}
    for k, v in get_real_person.items():
        point[k] = v

    POINTS[key] = point

# find a corresponding joint collection of the reference
reference_joints = POINTS[best_smash_name_only]
reference_neck = reference_joints['neck']
reference_hip = reference_joints['Lhip']

dest = []
for k,v in reference_joints.items():
    dest.append(v)

# Let's do a iterative point matching
for img in image_names:
    if(img != best_smash):
        
        filename = os.path.basename(img)
        filename = filename.split('_')
        filename = filename[0] + '_' +  filename[1] + '_' +  filename[2] + '_' +  filename[3]

        src_joints = POINTS[filename]
        src_neck = src_joints['neck']
        src_hip = src_joints['Lhip']

        neck_diff_x = reference_neck[0] - src_neck[0]
        neck_diff_y = reference_neck[1] - src_neck[1]
        
        #new_src_hip_x = src_hip[0] + neck_diff_x
        #new_src_hip_y = src_hip[1] + neck_diff_y

        # reference neck - reference hip = v1
        # reference neck - src_hip = v2
        v1 = np.array([reference_neck[0] - reference_hip[0], reference_neck[1] - reference_hip[1]])
        v2 = np.array([src_neck[0] - src_hip[0], src_neck[1] - src_hip[1]])

        th = np.dot(v1, v2)/(norm(v1)*norm(v2))
        th = np.arccos(th)

        init_pose = np.array([[np.cos(th), -np.sin(th), neck_diff_x], [np.sin(th), np.cos(th), neck_diff_y], [0, 0, 1]])
        
        src = []
        for k,v in src_joints.items():
            src.append(v)

        T, distances, iterations = icp.icp(np.asarray(src), np.asarray(dest), init_pose, tolerance=1e-200) # src, des
        
        row = label.loc[label[0] == (filename+'.png')]

        C = np.ones((len(src_joints),3))
        C[:,0:2] = np.copy(np.asarray(src))
        C = np.dot(T, C.T).T

        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)
        color = (B,G,R)

        for point in C:
            p = (int(point[0]), int(point[1]))
            print(p)
            reference = cv2.circle(reference, p, 2, color, 3)

        cv2.imshow('hi', reference)
        cv2.waitKey(0)
        


'''
# once we found the T -> transform matrix, we will need to transform bounding box coordinates
row = label.loc[label[0] == (filename+'.png')]

old_x_min = row[2].values[0]
old_y_min = row[3].values[0]
old_x_max = row[4].values[0]
old_y_max = row[5].values[0]

src_bbox = np.array([[row[2].values[0], row[3].values[0]], [row[4].values[0], row[5].values[0]]]) # xmin, ymin, xmax, ymax     
C = np.ones((2, 3))
C[:,0:2] = np.copy(src_bbox)
C = np.dot(T, C.T).T

new_x_min = int(C[0][0])
new_y_min = int(C[0][1])
new_x_max = int(C[1][0])
new_y_max = int(C[1][1])

src_img = cv2.imread(os.path.join(data_path, filename, '_result.png'))
crop_src_img = src_img[old_x_min:old_x_max, old_y_min:old_y_max]




print(C)
'''