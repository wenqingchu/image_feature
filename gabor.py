#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2



def build_filters():
    filters = []

    # size of Gabor kernel
    ksize = 31

    # for different orientations
    for theta in np.arange(0, np.pi, np.pi / 8):
        for lamda in np.arrange(0, np.pi, np.pi/4):
            kern.cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append((kern<Plug>PeepOpenarams))
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

gabor_feature = np
g = some_image
filters = build_filters()
filtered_images = process(g, filters)

# Step 1 - Obtain the set of bags of features.

height = 256
width = 256
dictionary_size = 1024
base_path = "img/all/apple/"
feature_path = "feature/sift/"
im_list_file = open("img/all/apple.txt", 'r')
im_list = []
for line in im_list_file:
    im_list.append(line.strip())


sift = cv2.SIFT()
bow_train = cv2.BOWKMeansTrainer(dictionary_size)



for i in range(len(im_list)):
    img = cv2.imread(base_path + im_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (height, width))
    kp, des = sift.detectAndCompute(img, None)
    bow_train.add(des)


dictionary = bow_train.cluster()

# step 2 - Obtain sift feature for each image

detect = cv2.FeatureDetector_create("SIFT")
extract = cv2.DescriptorExtractor_create("SIFT")
flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})
bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

bow_extract.setVocabulary(dictionary)

for i in range(len(im_list)):
    img = cv2.imread(base_path + im_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (height, width))
    output_feature = bow_extract.compute(img, detect.detect(img))
    img_name = im_list[i].split('/')[-1]
    img_name = img_name.split('.')[0]
    output_path = feature_path + img_name + ".npy"
    np.save(output_path, output_feature)







