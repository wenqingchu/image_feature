#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os
import sys


# Get the feature vector (local energy/mean amplitude from response matrices) of an image
def get_image_feature_vector(image):
    ret_feature = cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    ret_feature  = ret_feature / 256
    return ret_feature




height = 256
width = 256

feature_path = "feature/histogram/"
if os.path.exists(feature_path) == False:
    os.makedirs(feature_path)

image_classes = ['apple', 'pottery', 'other', 'glass']



for image_class in image_classes:
    base_path = "img/all/" + image_class + "/"
    output_feature_path = feature_path + image_class + "/"
    if os.path.exists(output_feature_path) == False:
        os.makedirs(output_feature_path)
    im_list_file = open("img/all/" + image_class + "/" + image_class + ".txt", 'r')
    im_list = []
    for line in im_list_file:
        im_list.append(line.strip())
    for i in range(len(im_list)):
        img = cv2.imread(im_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (height, width))
        output_feature = get_image_feature_vector(img)
        img_name = im_list[i].split('/')[-1]
        img_name = img_name.split('.')[0]
        output_path = output_feature_path + img_name + ".npy"
        np.save(output_path, output_feature)






