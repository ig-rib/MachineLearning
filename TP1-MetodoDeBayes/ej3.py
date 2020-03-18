#!/bin/python3

import os
import random
import pandas as pd

base = './Docu'
percentage = 0.7


# classes = os.listdir(base)
# filesPerClass = {}
# for c in classes:
#     classFiles = os.listdir(base + "/" + c)
#     random.shuffle(classFiles)
#     splittingIndex = int(len(classFiles)*percentage)
#     training = classFiles[:splittingIndex]
#     test = classFiles[splittingIndex:]
#     filesPerClass[c] = [training, test]

# print()

# data = pd.read_excel('./Noticias_argentinas.xlsx')
data = pd.read_csv('./aa_bayes.tsv', sep='\t')
data = data[( data['categoria'] != 'Destacadas') & (data['categoria'] != 'Noticias destacadas') & (data['categoria'] != 'NaN')]

wordCount = {}
categories = data.categoria.unique()
for cat in categories:
    wordCount[cat] = {}

for i in range(len(data)):
    for word in data.iloc[i]['titular']:
        # for cat in categories:
        #     if wordCount[cat].get(word, None) == None:
        #         wordCount[cat][word] = 0
        wordCount[data.iloc[i]['categoria']][word] = wordCount[data.iloc[i]['categoria']].get(word, 0) + 1

    

print()