import numpy as np

class ClusterNode:
    def __init__(self, p=None, nodes=[], children=None):
        self.p = None
        self.nodes = nodes
        self.children = children

    ## Get max distance between any 2 examples from
    # different clusters, between this and another
    # cluster 
    def getMaxDistance(self, other, matrix):
        return max([max([matrix.item((i, j)) for j in other.nodes]) for i in self.nodes])

    def __repr__(self):
        return f"ClusterNode {self.nodes}" # % (self.nodes)

    def __str__(self):
        return f"ClusterNode {self.nodes}" # % ()


class HierarchicalClustering:

    def __init__(self):
        ## Unimplemented
        self

    def group(self, D):
        groups = [ClusterNode(nodes=[i]) for i in range(len(D))]
        distMatrix = np.matrix([[np.linalg.norm(D[i] - D[j]) for j in range(len(D))] for i in range(len(D))])
        ## as long as there are more than one groups left
        while len(groups) > 1:
            distances = [ (x, min([(x.getMaxDistance(other, distMatrix), other) for other in groups if other != x], key=lambda x: x[0])) for x in groups]
            minPair = min(distances, key=lambda x: x[1][0])
            newNode = ClusterNode(children=[minPair[0], minPair[1][1]], nodes=[*minPair[0].nodes, *minPair[1][1].nodes])
            minPair[0].p = newNode
            minPair[1][1].p = newNode
            groups.remove(minPair[0])
            groups.remove(minPair[1][1])
            groups.append(newNode)
        
        return newNode

# D = np.matrix([
#     [0, 1, 2],
#     [3, 4, 5],
#     [10, 7, 8],
#     [1, 5, 2],
#     [10, 4, 1]
# ])

# distMatrix = [[np.linalg.norm(D[i] - D[j]) for j in range(len(D))] for i in range(len(D))]
# for i in range(len(D)):
#     print(distMatrix[i])

# print(np.linalg.norm(D[0] - D[1]))

# hc = HierarchicalClustering()
# hc.group(D)