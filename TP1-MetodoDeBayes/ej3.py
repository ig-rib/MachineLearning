#!/bin/python3

import os
import random

base = 'TP1-MetodoDeBayes/Docu'
percentage = 0.7


classes = os.listdir(base)
filesPerClass = {}
for c in classes:
    classFiles = os.listdir(base + "/" + c)
    random.shuffle(classFiles)
    splittingIndex = int(len(classFiles)*percentage)
    training = classFiles[:splittingIndex]
    test = classFiles[splittingIndex:]
    filesPerCalss[c] = [training, test]

print()