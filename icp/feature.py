import os
import cv2
import json
import pandas as pd
import numpy as np
from numpy.linalg import norm

def find_angle(v1, v2):
    th = np.dot(v1, v2)/(norm(v1)*norm(v2))
    if(th >= 1.0):
        th = 0.0
    else:
        th = np.arccos(th)

    return th

data_path = 'data/LCW/'
shot_type = ['smash', 'backhand']

best_smash = 'unlabelled_LCW_Japan2017_43'
best_backhand = 'unlabelled_LCW_Malaysia2017_20'

data_smash = ''
with open('LCW_smash.json', "r") as read_file:
    data_smash = json.load(read_file)

SMASH_POINTS = {}

for key in data_smash:
    # key is the filename
    large_list = data_smash[key]
    
    get_real_person = large_list[0]

    for item in large_list:
        if(len(item) > len(get_real_person)):
            get_real_person = item
    
    # iterate dictionary
    point = {}
    for k, v in get_real_person.items():
        point[k] = v

    SMASH_POINTS[key] = point

data_backhand = ''
with open('LCW_backhand.json', "r") as read_file:
    data_backhand = json.load(read_file)

BACKHAND_POINTS = {}

for key in data_backhand:
    # key is the filename
    large_list = data_backhand[key]
    
    get_real_person = large_list[0]

    for item in large_list:
        if(len(item) > len(get_real_person)):
            get_real_person = item
    
    # iterate dictionary
    point = {}
    for k, v in get_real_person.items():
        point[k] = v

    BACKHAND_POINTS[key] = point


best_smash_joints = SMASH_POINTS[best_smash]
best_backhand_joints = BACKHAND_POINTS[best_backhand]

print(best_smash_joints)
print(best_backhand_joints)

#####################SMASH######################
# LSHO, NECK, RSHO, RELB, RWRI
################################################
SMASH_FEATURE = []

smash_lsho = best_smash_joints['Lsho']
smash_neck = best_smash_joints['neck']
smash_rsho = best_smash_joints['Rsho']
smash_relb = best_smash_joints['Relb']
smash_rwri = best_smash_joints['Rwri']
print('lsho', smash_lsho)
print('neck', smash_neck)
print('rsho', smash_rsho)
print('relb', smash_relb)
print('rwri', smash_rwri)

smash_f1_v1 = np.array([smash_lsho[0] - smash_neck[0], smash_lsho[1] - smash_neck[1]])
smash_f1_v2 = np.array([smash_neck[0] - smash_rsho[0], smash_neck[1] - smash_rsho[1]])

print(smash_f1_v1)
print(smash_f1_v2)
feature1 = find_angle(smash_f1_v1, smash_f1_v2)
print(np.rad2deg(feature1))
SMASH_FEATURE.append(np.rad2deg(feature1))

smash_f2_v1 = np.array([smash_neck[0] - smash_rsho[0], smash_neck[1] - smash_rsho[1]])
smash_f2_v2 = np.array([smash_rsho[0] - smash_relb[0], smash_rsho[1] - smash_relb[1]])

print(smash_f2_v1)
print(smash_f2_v2)
feature2 = find_angle(smash_f2_v1, smash_f2_v2)
print(np.rad2deg(feature2))
SMASH_FEATURE.append(np.rad2deg(feature2))

smash_f3_v1 = np.array([smash_rsho[0] - smash_relb[0], smash_rsho[1] - smash_relb[1]])
smash_f3_v2 = np.array([smash_relb[0] - smash_rwri[0], smash_relb[1] - smash_rwri[1]])

print(smash_f3_v1)
print(smash_f3_v2)
feature3 = find_angle(smash_f3_v1, smash_f3_v2)
print(np.rad2deg(feature3))
SMASH_FEATURE.append(np.rad2deg(feature3))

#####################BACKHAND######################
# LSHO, NECK, RSHO, RELB, RWRI
###################################################
BACKHAND_FEATURE = []

backhand_lsho = best_backhand_joints['Lsho']
backhand_neck = best_backhand_joints['neck']
backhand_rsho = best_backhand_joints['Rsho']
backhand_relb = best_backhand_joints['Relb']
backhand_rwri = best_backhand_joints['Rwri']
print('lsho', backhand_lsho)
print('neck', backhand_neck)
print('rsho', backhand_rsho)
print('relb', backhand_relb)
print('rwri', backhand_rwri)

backhand_f1_v1 = np.array([backhand_lsho[0] - backhand_neck[0], backhand_lsho[1] - backhand_neck[1]])
backhand_f1_v2 = np.array([backhand_neck[0] - backhand_rsho[0], backhand_neck[1] - backhand_rsho[1]])

print(backhand_f1_v1)
print(backhand_f1_v2)
feature1 = find_angle(backhand_f1_v1, backhand_f1_v2)
print(np.rad2deg(feature1))
BACKHAND_FEATURE.append(np.rad2deg(feature1))

backhand_f2_v1 = np.array([backhand_neck[0] - backhand_rsho[0], backhand_neck[1] - backhand_rsho[1]])
backhand_f2_v2 = np.array([backhand_rsho[0] - backhand_relb[0], backhand_rsho[1] - backhand_relb[1]])

print(backhand_f2_v1)
print(backhand_f2_v2)
feature2 = find_angle(backhand_f2_v1, backhand_f2_v2)
print(np.rad2deg(feature2))
BACKHAND_FEATURE.append(np.rad2deg(feature2))

backhand_f3_v1 = np.array([backhand_rsho[0] - backhand_relb[0], backhand_rsho[1] - backhand_relb[1]])
backhand_f3_v2 = np.array([backhand_relb[0] - backhand_rwri[0], backhand_relb[1] - backhand_rwri[1]])

print(backhand_f3_v1)
print(backhand_f3_v2)
feature3 = find_angle(backhand_f3_v1, backhand_f3_v2)
print(np.rad2deg(feature3))
BACKHAND_FEATURE.append(np.rad2deg(feature3))

print(SMASH_FEATURE)
print(BACKHAND_FEATURE)