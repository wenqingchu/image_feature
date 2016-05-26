#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os
import sys


# Step 1 - Obtain the set of bags of features.

height = 256
width = 256
dictionary_size = 256

feature_path = "feature/sift/"
if os.path.exists(feature_path) == False:
    os.makedirs(feature_path)

sift = cv2.SIFT(0,3,0.02,10,1.6)
bow_train = cv2.BOWKMeansTrainer(dictionary_size)

image_classes = ['apple', 'pottery', 'other', 'glass']

im_list = []
for image_class in image_classes:
    im_list_file = open("img/all/" + image_class + "/" + image_class + ".txt", 'r')
    for line in im_list_file:
        im_path = line.strip()
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (height, width))
        kp, des = sift.detectAndCompute(img, None)
        print len(kp)
        bow_train.add(des)


dictionary = bow_train.cluster()

# step 2 - Obtain sift feature for each image

detect = cv2.FeatureDetector_create("SIFT")
extract = cv2.DescriptorExtractor_create("SIFT")
flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})
bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

bow_extract.setVocabulary(dictionary)
for image_class in image_classes:
    im_list_file = open("img/all/" + image_class + "/" + image_class + ".txt", 'r')
    output_feature_path = feature_path + image_class + "/"
    if os.path.exists(output_feature_path) == False:
        os.makedirs(output_feature_path)
    for line in im_list_file:
        im_path = line.strip()
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (height, width))
        output_feature = bow_extract.compute(img, sift.detect(img, None))[0]
        img_name = im_path.split('/')[-1]
        img_name = img_name.split('.')[0]
        output_path = output_feature_path + img_name + ".npy"
        np.save(output_path, output_feature)







