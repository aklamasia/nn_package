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


def clasifyClass(trainData,trainLabel,testDataInstance,k,nClasses):
    distances = []
    testMembership = [0.0]*nClasses
    inputAttr = len(testDataInstance)
    
    for x in range(len(trainData)):
        dist = euclideanDistance(testDataInstance, trainData[x], inputAttr)
        distances.append((trainData[x],x, dist))
        
    distances.sort(key=operator.itemgetter(2))
    
    neighbors = []
    minDist = []
    for x in range(k):
        neighbors.append(distances[x][1])
        minDist.append(distances[x][2])
 
        
    quality = 0
    
    R = [0]*k
    lower = [1.0]*nClasses
    upper = [0.0]*nClasses
    minimum = sys.float_info.max
    sumMinimum=0
    
    for l in range(k):
        for j in range(inputAttr):
            dist=1.0-abs(trainData[neighbors[l]][j]-testDataInstance[j])
            if minimum > dist:
                minimum=dist
        R[l]=minimum
    
        for c in range(nClasses):
            if c == trainLabel[neighbors[l]]:
                lower[c] = min(lower[c],max(1.0-R[l],1.0))
                upper[c] = max(upper[c],min(R[l],1.0))
            else:
               lower[c]=min(lower[c], max(1.0-R[l],0.0))
               upper[c]=max(upper[c], min(R[l],0.0))
               
    
    outputClass = -1
    
    for c in range(nClasses):
        if quality<(lower[c]+upper[c])/2.0:
            quality=(lower[c]+upper[c])/2.0
            outputClass=c
            
    return outputClass


def frnn_frs(trainData, trainLabel, testData, testLabel, k):
    labTrainSet, labTestSet, legend = remapLabels(rlabTrainSet=trainLabel, rlabTestSet=testLabel)
    nClasses = len(legend)
    index_testLabel = 0
    correct = 0
    t1 = time()
    for testDataInstance in testData:
        predict = clasifyClass(trainData,labTrainSet,testDataInstance,k,nClasses)
        progress(index_testLabel+1, len(testData))
        if legend[predict] == legend[labTestSet[index_testLabel]]:
            correct+=1
        index_testLabel+=1
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accouracy = round(((float(correct)/float(len(testData)))*100.0),2)
    return accouracy,(time() - t1)/len(testData)