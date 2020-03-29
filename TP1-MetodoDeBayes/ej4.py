import pandas as pd

data = pd.read_csv('binary.csv', sep=',')

classes = ['gre', 'admit', 'gpa', 'rank']
parents = {'gre': ['rank'], 'gpa': ['rank'], 'rank': [], 'admit': ['gpa', 'gre']}
children = {'gre': ['admit'], 'gpa': ['admit'], 'admit': [], 'rank': ['gpa', 'gre', 'admit']}
nodes = []

T = {}

def mapToCategory(x, limit):
    if x >= limit:
        return 1
    else:
        return 0

data['gre'] = data['gre'].map(lambda x: mapToCategory(x, 500))
data['gpa'] = data['gpa'].map(lambda x: mapToCategory(x, 3))
data['admit'] = data['admit'].map(lambda x: mapToCategory(x, 1))

def conditionalProbability(dataTable, target, targetValue, conditions, conditionValues):
    denom = dataTable
    for index, condition in enumerate(conditions):
        denom = denom[denom[condition] == conditionValues[index]]
    numerator = denom[denom[target] == targetValue]
    return len(numerator)/len(denom)

pA = {}
for i in range(1, 5):
    pA[i] = len(data[data['rank'] == i])/len(data)

print('a) P(admitido=0 | rango=1) = %g' % (conditionalProbability(data, 'admit', 0, ['rank'], [1])))
print('b) P(admitido=1 | rank=2, gre=450, gpa=3.5) = %g' % (conditionalProbability(data, 'admit', 1, ['rank', 'gre', 'gpa'], [2, 0, 1])))
print('c) El aprendizaje es paramétrico')