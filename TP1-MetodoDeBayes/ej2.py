#!/bin/python3

import pandas as pd

input = [1, 0, 1, 1, 0]

excelFile = pd.ExcelFile('./PreferenciasBritanicos.xlsx')
data = excelFile.parse(excelFile.sheet_names[0])
# initialize D
D = {}
keys = list(data.keys())
keys.pop()

for i in range(len(data)):
    group = data.iloc[i]["Nacionalidad"]
    if D.get(group) == None:
        D[group] = {}
    for key in keys:
        D[group][key] = D[group].get(key, 0) + data.iloc[i][key]
    D[group]["total"] = D[group].get("total", 0) + 1
for group in D.keys():
    for key in keys:
        # Laplace Smoothing when calculating relative frequencies
        D[group][key] = (D[group][key]+1)/(D[group]["total"]+len(D.keys()))
inputHash = {}
for i in range(len(input)):
    inputHash[keys[i]] = input[i]

max = 0
maxNat = ''

aPosterioriProbs = {}

for group in D.keys():
    prob = 1
    for key in keys:
        if inputHash[key] == 1:
            prob *= D[group][key]
        else:
            prob *= (1-D[group][key])
    prob *= (D[group]["total"]/len(data))
    aPosterioriProbs[group] = prob
    if prob > max:
        max = prob
        maxNat = group
print('\nVector de entrada ', input, ' clasificado como ', maxNat)
for nat in aPosterioriProbs.keys():
    print('Probabilidad a posteriori (sin denominador) para %s: ' % (nat), aPosterioriProbs[nat])