import os
import numpy as np
from numpy.linalg import norm

def find_angle(v1, v2):
    th = np.dot(v1, v2)/(norm(v1)*norm(v2))
    if(th >= 1.0):
        th = 0.0
    else:
        th = np.arccos(th)

    return th

def calculate_angle(joint_record):
    lsho = joint_record['Lsho']
    neck = joint_record['neck']
    rsho = joint_record['Rsho']
    relb = joint_record['Relb']
    rwri = joint_record['Rwri']

    f1_v1 = np.array([lsho[0] - neck[0], lsho[1] - neck[1]])
    f1_v2 = np.array([neck[0] - rsho[0], neck[1] - rsho[1]])
    
    feature1 = np.rad2deg(find_angle(f1_v1, f1_v2))

    f2_v1 = np.array([neck[0] - rsho[0], neck[1] - rsho[1]])
    f2_v2 = np.array([rsho[0] - relb[0], rsho[1] - relb[1]])
    feature2 = np.rad2deg(find_angle(f2_v1, f2_v2))

    f3_v1 = np.array([rsho[0] - relb[0], rsho[1] - relb[1]])
    f3_v2 = np.array([relb[0] - rwri[0], relb[1] - rwri[1]])
    feature3 = np.rad2deg(find_angle(f3_v1, f3_v2))

    return (feature1, feature2, feature3)