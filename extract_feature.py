# -*- coding: UTF-8 -*-
#!/usr/bin/python

import sys
import numpy as np
import cv2

import feature

def extract_feature(img, feature_name):
    if feature_name == "lbp":
        feature_function = feature.LbpFeature()
        ret_feature = feature_function.extract(img)
    elif feature_name == "sift":
        feature_function = feature.SiftFeature()
        ret_feature = feature_function.extract(img)
    elif feature_name == "gabor":
        feature_function = feature.GaborFeature()
        ret_feature = feature_function.extract(img)
    elif feature_name == "histogram":
        ret_feature = hist= cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    return ret_feature




feature_names = ["lbp", "sift", "gabor", "histogram"]

height = 256
width = 256
img_path = "707.jpg"
feature_path = "feature/"
img_name  =  img_path.split('/')[-1]
img_name = img_name.split('.')[0]


if os.path.exists(feature_path) == False:
    os.makedirs(feature_path)




for feature_name in feature_names:
    output_path = feature_path + feature_name + "/"
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (height, width))
        output_feature = extract_feature(img, feature_name)
        output_filename = output_path+img_name+".npy"
        np.save(output_filename, output_feature)


