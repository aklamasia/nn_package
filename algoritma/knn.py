import math
import operator
from time import time, strftime
import sys


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

 
def getNeighbors(trainData, trainLabel, testInstance, k ):
    distances = []
    length = len(testInstance)
    for x in range(len(trainData)):
        dist = euclideanDistance(testInstance, trainData[x], length)
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

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s/%s\r' % (bar, percents, '%', str(count), str(total)))
    sys.stdout.flush()

def knn(trainData, trainLabel, testData, testLabel,k):
    predictions=[]
    counter = 1
    t1 = time()
    log = []
    
    for x in range(len(testData)):
        temp = []
        t0 = time()
        neighbors = getNeighbors(trainData, trainLabel, testData[x], k)
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
        counter+=1
        
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accuracy = getAccuracy(testLabel, predictions)
    return accuracy, (time() - t1)/len(testData)