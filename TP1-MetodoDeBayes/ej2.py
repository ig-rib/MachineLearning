#!/bin/python3

import pandas as pd

input = [1, 0, 1, 1, 0]

excelFile = pd.ExcelFile('./TP1-MetodoDeBayes/PreferenciasBritanicos.xlsx')
data = excelFile.parse(excelFile.sheet_names[0])
print(data)
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
        D[group][key] = (D[group][key]+1)/(D[group]["total"]+len(D.keys()))
inputHash = {}
for i in range(len(input)):
    inputHash[keys[i]] = input[i]

max = 0
maxNat = ''
for group in D.keys():
    prob = 1
    for key in keys:
        if inputHash[key] == 1:
            prob *= D[group][key]
        else:
            prob *= (1-D[group][key])
    prob *= (len(D[group])/len(data))
    if prob > max:
        max = prob
        maxNat = group
print(maxNat, max)