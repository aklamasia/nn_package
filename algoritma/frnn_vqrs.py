import operator
from time import time, strftime
from itertools import groupby
import math
import operator
import sys
from utils import progress


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def remapLabels(rlabTrainSet=[],rlabTestSet=[] ):
    legendLabels = list(set(rlabTrainSet))
    
    templabTraingSet = []
    for x in rlabTrainSet:
        for y in range(len(legendLabels)):
            if x == legendLabels[y]:
                x = (y)
                templabTraingSet.append(x)
               
    templabTestSet = []
    for x in rlabTestSet:
        for y in range(len(legendLabels)):
            if x == legendLabels[y]:
                x = (y)
                templabTestSet.append(x)
    
    return templabTraingSet, templabTestSet, legendLabels


def VQ1(value):
    alpha = 0.1
    beta = 0.6
    if value<=alpha:
        return 0.0
    if value<=(alpha+beta)/2.0:
        return (2.0*((value-alpha)*(value-alpha)))/((beta-alpha)*(beta-alpha))
    if value<=beta:
        return 1.0-(2.0*((value-alpha)*(value-alpha)))/((beta-alpha)*(beta-alpha))
    return 1.0

def VQ2(value):
    alpha = 0.2
    beta = 1.0
    if value<=alpha:
        return 0.0
    if value<=(alpha+beta)/2.0:
        return (2.0*((value-alpha)*(value-alpha)))/((beta-alpha)*(beta-alpha))
    if value<=beta:
        return 1.0-(2.0*((value-alpha)*(value-alpha)))/((beta-alpha)*(beta-alpha))
    return 1.0

def clasifyClass(trainData,trainLabel,testDataInstance,k,nClasses):
    distances = []
    testMembership = [0.0]*nClasses
    length = len(testDataInstance)
    for x in range(len(trainData)):
        dist = euclideanDistance(testDataInstance, trainData[x], length)
        distances.append((trainData[x],x, dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = []
    minDist = []
    for x in range(k):
        neighbors.append(distances[x][1])
        minDist.append(distances[x][2])
        
    quality = 0
    R = [0]*k
    lower = [0]*nClasses
    upper = [0]*nClasses
    minimum = sys.float_info.max
    sumMinimum=0
    for x in range(k):
        for y in range(length):
            dist=1.0-abs(trainData[neighbors[x]][y]-testDataInstance[y])
            if minimum > dist:
                minimum=dist
        R[x]=minimum
        sumMinimum+=minimum
    
    for x in range(k):
        quality = VQ1(R[x]/sumMinimum)
        lower[trainLabel[neighbors[x]]] = max(lower[trainLabel[neighbors[x]]],quality)
        quality = VQ2(R[x]/sumMinimum)
        upper[trainLabel[neighbors[x]]] = max(upper[trainLabel[neighbors[x]]],quality)
    
    outputClass = -1
 
    for c in range(nClasses):
        if quality<=(lower[c]+upper[c])/2.0:
            quality=(lower[c]+upper[c])/2.0
            outputClass=c
            
    return outputClass

def frnn_vqrs(trainData, trainLabel, testData, testLabel,k):
    labTrainSet, labTestSet, legend = remapLabels(rlabTrainSet=trainLabel, rlabTestSet=testLabel)
    nClasses = len(legend)
    index_testLabel = 0
    t1 = time()
    correct =0
    for testDataInstance in testData:
        predict = clasifyClass(trainData,labTrainSet,testDataInstance,k,nClasses)
        progress(index_testLabel+1, len(testData))
        if legend[predict] == legend[labTestSet[index_testLabel]]:
            correct+=1
        index_testLabel+=1
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accouracy = round(((float(correct)/float(len(testData)))*100.0),2)
    return accouracy,(time() - t1)/len(testData)