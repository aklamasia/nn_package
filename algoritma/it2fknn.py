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
                x = (y+1)
                templabTraingSet.append(x)
               
    templabTestSet = []
    for x in rlabTestSet:
        for y in range(len(legendLabels)):
            if x == legendLabels[y]:
                x = (y+1)
                templabTestSet.append(x)
    
    return templabTraingSet, templabTestSet, legendLabels

def getNeighbors(trainData, index, k):
    distances = []
    length = len(trainData[index])-1
    for x in range(len(trainData)):
        if x !=index :
            dist = euclideanDistance(trainData[index],trainData[x], length)
            distances.append((trainData[x],dist,x))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    idx_neighbors = []
    for x in range(k):
        neighbors.append(distances[x][1])
        idx_neighbors.append(distances[x][2])
    return idx_neighbors,neighbors

 

def assignMembership(trainData, trainLabel,kMax, nClasses):
    kMax=3
    count = 0.0
    membership = [[0]*nClasses]*len(trainData)
    tempMem = []
    k=1
    i = 0
    while k <= kMax:
        temp = []
        for instance in range(len(trainData)):
            mindist = []
            nearestN = []
            instance_member = [0]*nClasses
            selectedClasses = [0]*nClasses
                
            nearestN, mindist = getNeighbors(trainData, instance, k)
            
            for x in range(k):
                selectedClasses[trainLabel[nearestN[x]]-1]+=1
            
            for x in range(nClasses):
                term = float(selectedClasses[x])/float(k)
                if trainLabel[instance]-1==x:
                    instance_member[x]+=0.51+0.49*term
                else:
                    instance_member[x]+=0.49*term
                    
            i+=1
            progress(i,len(trainData)*(kMax/2+1))
            temp.append(instance_member)
        tempMem.append(temp)
        count+=1.0
        k+=2
 
    for i in range((kMax/2)+1):
        for x in range(len(trainData)):
            membership[x]=map(operator.add, membership[x],tempMem[i][x])
        
    for i in range(len(trainData)):
        for j in range(nClasses):
            membership[i][j]/=count
    
    return membership

def computeMembership(testDataInstance,trainData, membership, k,m, nClasses):
    
    #KNN
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
    
    norm = [0.0]*k
    sumNorm=0.0
    for x in range(k):
        if(minDist[x]==0.0):
            norm[x]= 1
        else:
            norm[x]= pow(minDist[x],(-2/m-1))
        sumNorm+=norm[x]
    
    for x in range(k):
        for y in range(nClasses): 
            testMembership[y]+=membership[neighbors[x]][y]*(norm[x]/sumNorm)
    
    return testMembership

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s/%s\r' % (bar, percents, '%', str(count), str(total)))
    sys.stdout.flush()

def computeClass(testMembership):
    max_index, max_value = max(enumerate(testMembership), key=operator.itemgetter(1))
    return max_index       

def it2fknn(trainData, trainLabel, testData, testLabel,k):
    membership = []
    referenceMembership = []
    testMembership = []
    m=2
    
   
    labTrainSet, labTestSet, legend = remapLabels(rlabTrainSet=trainLabel, rlabTestSet=testLabel)
    nClasses = len(legend)
    t0 = time()
    membership = assignMembership(trainData,labTrainSet,k,nClasses)
    timeextract= ("%0.5fs"%((time() - t0)))
    print "assign membership >" + timeextract
    correct = 0
    t1 = time()
    for x in range(len(testData)):
        testMembership = computeMembership(testData[x],trainData, membership,k,m, nClasses)
        predict = computeClass(testMembership)
        progress(x+1, len(testData))
        if legend[predict] == legend[labTestSet[x]-1]:
            correct+=1
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accouracy = round(((float(correct)/float(len(testData)))*100.0),2)
    return  accouracy,(time() - t1)/len(testData), timeextract