import operator
from time import time, strftime
from itertools import groupby
import math
import operator
import sys


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

def computeMembership(trainData, trainLabel, nClasses, k):
    membership = [[0.0]*nClasses]*len(trainData)

    distances = []
    testMembership = [0.0]*nClasses
    for instance in range(len(trainData)):
        progress(instance+1, len(trainData))
        length = len(trainData[instance])
        for i in range(len(trainData)):
            if i!=instance:
                dist = euclideanDistance(trainData[i], trainData[instance], length)
                distances.append((trainData[i],i, dist))
        distances.sort(key=operator.itemgetter(2))
        selectedClasses = [0]*nClasses
        
        for x in range(k):
            selectedClasses[trainLabel[distances[x][1]]]+=1
    
        for x in range(nClasses):
            term = float(selectedClasses[x]) / float(k)
            if trainLabel[instance]==x:
                membership[instance][x]=0.51+0.49*term
            else:
                membership[instance][x]=0.49*term
    return membership
        
    
def computeClass(member):
    out = 0
    maxi = member[0]
    
    for i in range(member):
        if(maxi<member[i]):
            maxi=member[i];
            out = i
   
    return out

def computeVote(value):
    mem = value
    nonMem = 1.0-value
    
    mA=0.49
    mR=0.21
    vA=0.51
    vR=0.79
    
    if (mem>=mA) and (nonMem<=vA):
        return mem
    if (mem<mR) and (nomMem>vR):
        return (-1.0*nonMem)
    return 0.0

def clasifyClass(trainData,trainLabel,testDataInstance,k,nClasses,membership):
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
        
    for i in range(k):
        for j in range(nClasses):
            testMembership[j]+=computeVote(membership[neighbors[i]][j])
    
    for i in range(nClasses):
        testMembership[i]=0.5*(1.0+(testMembership[c]/float(k)))
    
    result = computeClass(testMembership)

    return result

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s/%s\r' % (bar, percents, '%', str(count), str(total)))
    sys.stdout.flush()

def if_knn(trainData, trainLabel, testData, testLabel,k):
    labTrainSet, labTestSet, legend = remapLabels(rlabTrainSet=trainLabel, rlabTestSet=testLabel)
    nClasses = len(legend)
    t0 = time()
    membership = computeMembership(trainData, labTrainSet, nClasses, k)
    timeextract= ("%0.5fs"%((time() - t0)))
    print "assign membership >" + timeextract
    index_testLabel = 0
    t1 = time()
    correct =0
    for testDataInstance in testData:
        predict = clasifyClass(trainData,labTrainSet,testDataInstance,k,nClasses,membership)
        progress(index_testLabel+1, len(testData))
        if legend[predict] == legend[labTestSet[index_testLabel]]:
            correct+=1
        index_testLabel+=1
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accouracy = round(((float(correct)/float(len(testData)))*100.0),2)
    return accouracy,(time() - t1)/len(testData), timeextract