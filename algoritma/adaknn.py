import math
import operator
from time import time, strftime
import sys
from utils import progress

def calculateRadius(trainData,trainLabel):
    radius =[]
    for indexTraining1 in range(len(trainData)):
        ownClass = trainLabel[indexTraining1]
        minDist = sys.float_info.max
        # progress(indexTraining1+1, len(trainData))
        for indexTraining2 in range(len(trainData)):
            if ownClass != trainLabel[indexTraining2]:
                length = len(trainData[indexTraining2])
                distance = euclideanDistance(trainData[indexTraining1], trainData[indexTraining2], length)
                if (distance<minDist):
                    minDist=distance
        radius.append(minDist)
    return radius

def adaptiveDistance(instance1, instance2, length, radius):
    distance = 0
    distance = euclideanDistance(instance1, instance2, length)
    distance = distance/radius
    return distance


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

 
def getNeighbors(trainData,trainLabel,testInstance, k, radius):
    distances = []
    length = len(testInstance)
    for x in range(len(trainData)):
        dist = adaptiveDistance(testInstance, trainData[x], length, radius[x])
        distances.append((trainLabel[x], dist,x))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testLabel, predictions):
    correct = 0
    for x in range(len(testLabel)):
        if testLabel[x] == predictions[x]:
            correct += 1
    return round((correct/float(len(testLabel))) * 100.00, 2)


def adaknn(trainData, trainLabel, testData, testLabel,k):
    predictions=[]
    log = []
    
    t3 = time()
    radius = calculateRadius(trainData,trainLabel)
    time_radius= ("%0.3fs"%(time() - t3))
    # print("All calculate radius done in %s" % time_radius)
    
    t1 = time()
    for x in range(len(testData)):
        temp = []
        t0 = time()
        neighbors = getNeighbors(trainData,trainLabel,testData[x], k, radius)
        result = getResponse(neighbors)
        predictions.append(result)
        progress(x+1, len(testData))
        timepredict= ("%0.3fs"%(time() - t0))
        temp.append(testLabel[x])
        temp.append(result)
        if testLabel[x] == result:
            temp.append("1")
        else:
            temp.append("0")
        log.append(temp)
      
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accuracy = getAccuracy(testLabel, predictions)
    return accuracy, (time() - t1)/len(testData), time_radius