#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os
import sys
from scipy.signal import convolve2d



class LocalDescriptor(object):
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def __call__(self,X):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")

    @property
    def neighbors(self):
        return self._neighbors

    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % (self._neighbors)

class OriginalLBP(LocalDescriptor):
    def __init__(self):
        LocalDescriptor.__init__(self, neighbors=8)

    def __call__(self,X):
        X = np.asarray(X)
        X = (1<<7) * (X[0:-2,0:-2] >= X[1:-1,1:-1]) \
            + (1<<6) * (X[0:-2,1:-1] >= X[1:-1,1:-1]) \
            + (1<<5) * (X[0:-2,2:] >= X[1:-1,1:-1]) \
            + (1<<4) * (X[1:-1,2:] >= X[1:-1,1:-1]) \
            + (1<<3) * (X[2:,2:] >= X[1:-1,1:-1]) \
            + (1<<2) * (X[2:,1:-1] >= X[1:-1,1:-1]) \
            + (1<<1) * (X[2:,:-2] >= X[1:-1,1:-1]) \
            + (1<<0) * (X[1:-1,:-2] >= X[1:-1,1:-1])
        return X

    def __repr__(self):
        return "OriginalLBP (neighbors=%s)" % (self._neighbors)


class ExtendedLBP(LocalDescriptor):
    def __init__(self, radius=1, neighbors=8):
        LocalDescriptor.__init__(self, neighbors=neighbors)
        self._radius = radius

    def __call__(self,X):
        X = np.asanyarray(X)
        ysize, xsize = X.shape
        # define circle
        angles = 2*np.pi/self._neighbors
        theta = np.arange(0,2*np.pi,angles)
        # calculate sample points on circle with radius
        sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
        sample_points *= self._radius
        # find boundaries of the sample points
        miny=min(sample_points[:,0])
        maxy=max(sample_points[:,0])
        minx=min(sample_points[:,1])
        maxx=max(sample_points[:,1])
        # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
        blocksizey = np.ceil(max(maxy,0)) - np.floor(min(miny,0)) + 1
        blocksizex = np.ceil(max(maxx,0)) - np.floor(min(minx,0)) + 1
        # coordinates of origin (0,0) in the block
        origy =  0 - np.floor(min(miny,0))
        origx =  0 - np.floor(min(minx,0))
        # calculate output image size
        dx = xsize - blocksizex + 1
        dy = ysize - blocksizey + 1
        # get center points
        C = np.asarray(X[origy:origy+dy,origx:origx+dx], dtype=np.uint8)
        result = np.zeros((dy,dx), dtype=np.uint32)
        for i,p in enumerate(sample_points):
            # get coordinate in the block
            y,x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and y.
            fx = np.floor(x)
            fy = np.floor(y)
            cx = np.ceil(x)
            cy = np.ceil(y)
            # calculate fractional part
            ty = y - fy
            tx = x - fx
            # calculate interpolation weights
            w1 = (1 - tx) * (1 - ty)
            w2 =      tx  * (1 - ty)
            w3 = (1 - tx) *      ty
            w4 =      tx  *      ty
            # calculate interpolated image
            N = w1*X[fy:fy+dy,fx:fx+dx]
            N += w2*X[fy:fy+dy,cx:cx+dx]
            N += w3*X[cy:cy+dy,fx:fx+dx]
            N += w4*X[cy:cy+dy,cx:cx+dx]
            # update LBP codes
            D = N >= C
            result += (1<<i)*D
        return result

    @property
    def radius(self):
        return self._radius

    def __repr__(self):
        return "ExtendedLBP (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)


class AbstractFeature(object):

    def compute(self,X,y):
        raise NotImplementedError("Every AbstractFeature must implement the compute method.")

    def extract(self,X):
        raise NotImplementedError("Every AbstractFeature must implement the extract method.")

    def save(self):
        raise NotImplementedError("Not implemented yet (TODO).")

    def load(self):
        raise NotImplementedError("Not implemented yet (TODO).")

    def __repr__(self):
        return "AbstractFeature"

class LbpFeature(AbstractFeature):
    def __init__(self, lbp_operator=OriginalLBP(), sz = (8,8)):
        AbstractFeature.__init__(self)
        if not isinstance(lbp_operator, LocalDescriptor):
            raise TypeError("Only an operator of type facerec.lbp.LocalDescriptor is a valid lbp_operator.")
        self.lbp_operator = lbp_operator
        self.sz = sz

    def compute(self,X,y):
        features = []
        for x in X:
            x = np.asarray(x)
            h = self.spatially_enhanced_histogram(x)
            features.append(h)
        return features

    def extract(self,X):
        X = np.asarray(X)
        return self.spatially_enhanced_histogram(X)

    def spatially_enhanced_histogram(self, X):
        # calculate the LBP image
        L = self.lbp_operator(X)
        # calculate the grid geometry
        lbp_height, lbp_width = L.shape
        grid_rows, grid_cols = self.sz
        py = int(np.floor(lbp_height/grid_rows))
        px = int(np.floor(lbp_width/grid_cols))
        E = []
        for row in range(0,grid_rows):
            for col in range(0,grid_cols):
                C = L[row*py:(row+1)*py,col*px:(col+1)*px]
                H = np.histogram(C, bins=2**self.lbp_operator.neighbors, range=(0, 2**self.lbp_operator.neighbors), normed=True)[0]
                # probably useful to apply a mapping?
                E.extend(H)
        return np.asarray(E)

    def __repr__(self):
        return "SpatialHistogram (operator=%s, grid=%s)" % (repr(self.lbp_operator), str(self.sz))



# Get the feature vector of an image
def get_image_feature_vector(image):
    feature_function = LbpFeature()
    return feature_function.extract(img)


height = 256
width = 256

feature_path = "feature/lbp/"
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







