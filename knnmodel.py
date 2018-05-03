import numpy as np
from six.moves import urllib
import operator
from datetime import datetime


def kNNClassify(newInput, dataSet, labels, k,p):
    p = float(p)
    numSamples = dataSet.shape[0] 
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
  
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    #print(diff.min(),diff.max())
    squaredDiff = diff ** p 
    squaredDist = np.sum(squaredDiff, axis = 1) 
    distance = squaredDist ** (1/p)
    sortedDistIndices = np.argsort(distance)

    classCount = {} 
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex






