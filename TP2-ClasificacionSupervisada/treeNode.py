#!/bin/python3

from collections import deque

class Node:
    def __init__(self, parent = None):
        self.filters = []
        self.children = {}
        self.parent = parent
        self.value = None
        self.height = None
    def getNodesAsList(self):
        stack = []
        finalList = []
        stack.append(self)
        while len(stack) > 0:
            curr = stack.pop()
            finalList.append(curr)
            stack.extend(curr.children.values())
        return finalList
    def toNOrLessNodes(self, N):
        nodes = sorted(deque(self.getNodesAsList()), key=lambda x: x.height)
        candidates = list(filter(lambda x: len(x.children.keys()) > 0, nodes))
        while len(nodes) > N and len(candidates) > 0:
            curr = candidates.pop()
            D = {}
            for child in curr.children.values():
                if D.get(child.value) == None:
                    D[child.value] = 0
                D[child.value] += 1
                nodes.remove(child)
            curr.children = {}
            curr.value = max(D, key=lambda x: D[x])
    def nodeCount(self):
        if len(self.children.values()) == 0:
            return 1
        return sum([ x.nodeCount() for x in self.children.values() ])