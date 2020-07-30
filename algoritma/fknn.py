import operator
from time import time, strftime
from itertools import groupby
import math
import operator
from utils import progress

def remapLabels(rlabTrainSet=[],rlabTestSet=[] ):
    legendLabels = list(set(rlabTrainSet))
    
    templabTraingSet = []
    for x in rlabTrainSet:
        for y in range(len(legendLabels)):
            if x == legendLabels[y]:
                x = (y+1)
                templabTraingSet.append(x)
               
    templabTestSet = []
    for x in rlabTestSet:
        for y in range(len(legendLabels)):
            if x == legendLabels[y]:
                x = (y+1)
                templabTestSet.append(x)
    
    return templabTraingSet, templabTestSet, legendLabels

def unaryMembership(ulabTrainSet=[], legendLabels=[]):
   
    max_class = len(legendLabels)
    buckets = [[0 for col in range(max_class)] for row in range(len(ulabTrainSet))]
    counter = 0
    for x in ulabTrainSet: 
        for y in range(max_class):
            if x-1 == y:
                buckets[counter][y]=1
        counter+=1
                
    return buckets

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k, unary ):
    distances = []
    m=2
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # print (testInstance,trainingSet[x],dist)
        distances.append(dist)
    neighbors_index = sorted(range(len(distances)),key=distances.__getitem__)
    neighbors_index  = [x+1 for x in neighbors_index]
    distances.sort()
    # print distances[:3]
    neighbors = []
    idx_neighbor = []
    # print neighbors_index
    for x in range(k):
        idx_neighbor.append(neighbors_index[x])
        if distances[x] == 0:
            # print "================="
            weight = 1
        else:
            weight = pow(distances[x],(-2/m-1))
        
        neighbors.append(weight)
    sum_neighbors = sum(neighbors)
    # print idx_neighbor
    # print neighbors
    test_out = [0 for col in range(len(unary[0]))]
    idx_test =[]
    
    for x in idx_neighbor:
        counter = 0
        for y in unary[x-1]:
            if y == 1:
                idx_test.append(counter)
            counter+=1
    
    # print idx_test   
    count = 0
    for x in idx_test:
        for y in range(len(test_out)):
            if x == y :
                test_out[y] = test_out[y]+neighbors[count]
                count+=1
    # print test_out
    test_out  = [x/sum_neighbors for x in test_out]
    # print test_out
    max_index, max_value = max(enumerate(test_out), key=operator.itemgetter(1))
    
    return max_index

def rematch_label(max_index,legend):
    for x in range(len(legend)):
        if x == max_index:
            predict_label = legend[x]
    return predict_label

def getAccuracy(labTestSet, predictions):
    correct = 0
    for x in range(len(labTestSet)):
        if labTestSet[x] == predictions[x]:
            correct += 1
    return round((correct/float(len(labTestSet))) * 100.00, 2) 
    

def fknn(trainingSet, labTrainSet, testSet, labTestSet, k):
    tempLabTestSet = labTestSet
    
    labTrainSet, labTestSet, legend = remapLabels(rlabTrainSet=labTrainSet, rlabTestSet=labTestSet)
    unary = unaryMembership(ulabTrainSet = labTrainSet, legendLabels=legend)
    
    predictions=[]
    t0 = time()
    for x in range(len(testSet)):
        temp = []
        
        neighbors = getNeighbors(trainingSet, testSet[x], k, unary)
        result = rematch_label(neighbors, legend)
        progress(x+1, len(testSet))
        predictions.append(result)
       
    timepredict= ("%0.5fs"%((time() - t0)/len(testSet)))
    
    accuracy = getAccuracy(tempLabTestSet, predictions)
   
    return accuracy, (time() - t0)/len(testSet)
