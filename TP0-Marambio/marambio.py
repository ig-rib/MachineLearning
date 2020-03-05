#!/bin/python3

import numpy as np
import pandas as pd

# data = np.loadtxt('marambio_2007.dat')
data = pd.read_csv('marambio_2007.dat', sep=' ', header='infer', engine="python")

#1 Muestre el nombre de las variables
results = {}

print("Variables: ")

for header in data.columns.values:

    dataList = list(data[header])
    results[header] = [ np.nanmin(dataList), 
    np.nanmax(dataList), 
    np.nanmean(dataList), 
    np.nanmedian(dataList), 
    np.nanpercentile(dataList, .25),
    np.nanpercentile(dataList, .75),
    ]

print("#1 Muestre el nombre de las variables:\n", ', '.join(list(data)))
print("#2 ¿Para qué período se registraron esos pronósticos?\n", "Para el período entre el 1/10/2007 y el 31/12/2007")
print("#3 Calcule la temperatura promedio de ese período para cadauno de los modelos climáticos involucrados")
for header in data.columns.values:
    print("{}: {}".format(header, results[header][2]))
print("#4 Calcular temperaturas máximas y mínimas para el modelo CMAM:")
print("Max: {}\nMin: {}".format(results["cmam"][0], results["cmam"][1]))
print("#5 Calcular medianas de las variables CMAM y UKMO:")
for header in data.columns.values:
    print("{}: {}".format(header, results[header][3]))
print("#6 Calcular medianas de las variables CMAM y UKMO:")
statistics = [ "Mínimo", "Máximo", "Media", "Mediana", "Q1", "Q3" ]
print("\t{}".format("\t".join(statistics)))
for header in data.columns.values:
    print("{}\t{}".format(header, "\t".join(map(lambda x: str(round(x, 3)), results[header]))))
print("#7 Dividir aleatoriamente el conjunto de registros en dos, dando como dato de entrada el porcentaje de elementos")
print("preguntar")