#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os
import sys

def build_filters():
    filters = []

    # size of Gabor kernel
    ksize = 31

    # for different orientations
    for theta in np.arange(0, np.pi, np.pi / 8):
        # different wavelengths of the sinusoidal factor
        for lamda in np.arange(0, np.pi, np.pi/4):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

# given an image and a set of filters, derive the response matrices
def process(img, filters):
    responses = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        responses.append(fimg)
    return responses


# Given a response matrix, compute for the local energy as a 8*8 matrix
# Local Energy = summing up the squared value of each matrix value from a response matrix
def get_local_energy (matrix):
    ret_feature = []
    box_w = len(matrix) / 8
    box_h = len(matrix) / 8
    for i in range(8):
        for j in range(8):
            local_energy = 0.0
            for row in range (box_w):
                for col in range(box_h):
                    cur_i = i * box_w + row
                    cur_j = j * box_h + col
                    val = int(matrix[cur_i][cur_j]) * int(matrix[cur_i][cur_j])
                    local_energy = local_energy + val
            # Divide by the highest possible value, which is 255^2 * (box_w x box_h)
            # to normalize values from 0 to 1, and replace 0s with EPS value to work with NB
            local_energy = local_energy / 65025 / box_w / box_h
            ret_feature.append(local_energy)
    return ret_feature

# Given a response matrix, compute for the mean amplitude as a 8*8 matrix
# Mean Amplitude = sum of absolute values of each matrix value from a response matrix
def get_mean_amplitude (matrix):
    ret_feature = []
    box_w = len(matrix) / 8
    box_h = len(matrix) / 8
    for i in range(8):
        for j in range(8):
            mean_amp = 0.0
            for row in range (box_w):
                for col in range(box_h):
                    cur_i = i * box_w + row
                    cur_j = j * box_h + col
                    val = abs(int(matrix[cur_i][cur_j]))
                    mean_amp = mean_amp + val
            # Divide by the highest possible value, which is 255 * (box-w x box_h)
            # to normalize values from 0 to 1, and replace 0s with EPS value to work with NB
            mean_amp = mean_amp / 255 / box_w / box_h
            ret_feature.append(mean_amp)
    return ret_feature



# Get the feature vector (local energy/mean amplitude from response matrices) of an image
def get_image_feature_vector(image,filters):
    ret_feature = []
    response_matrices = process(image, filters)
    for matrix in response_matrices:
        local_energy = get_local_energy(matrix)
        mean_amplitude = get_mean_amplitude(matrix)
        ret_feature.extend(local_energy)
        ret_feature.extend(mean_amplitude)
    return ret_feature



filters = build_filters()

height = 256
width = 256

feature_path = "feature/gabor/"
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
        output_feature = get_image_feature_vector(img, filters)
        img_name = im_list[i].split('/')[-1]
        img_name = img_name.split('.')[0]
        output_path = output_feature_path + img_name + ".npy"
        np.save(output_path, output_feature)







