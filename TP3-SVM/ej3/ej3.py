#!/bin/python3

import os
from scipy.misc import imread

dataPath = 'TP3-SVM/data/'

for image in [ f for f in os.listdir(dataPath) if '.jpg' in f ]:
    imArray = imread(dataPath + image)
    imArray
