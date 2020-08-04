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

def computeKTest(trainData,testData,Q, inputAtt):
    
    GlobalK = [0]*inputAtt
    exp=2.0/(Q-1.0)
    for j in range(inputAtt):
        sums=0.0
        for i in range(len(trainData)):
            dist=testData[j]-trainData[i][j]
            dist = pow(dist,exp)
            sums+=dist
        if sums!=0:
            GlobalK[j]=(len(trainData)-1)/(2.0*sums)
        else:
            GlobalK[j]=0
    return GlobalK


def clasifyClass(trainData,trainLabel,testDataInstance,index,k,nClasses):
    inputAtt = len(trainData[0])
    instance =[0]*inputAtt
    classPosibility=[0]*nClasses
    Q = 2.0
    GlobalK=computeKTest(trainData,testDataInstance, Q, inputAtt)
    
    for i in range(len(trainData)):
        distance = 0.0
        if i!=index:
            for j in range(inputAtt):
                distance+= GlobalK[j]*(testDataInstance[j]-trainData[i][j])*(testDataInstance[j]-trainData[i][j])
            
            classPosibility[trainLabel[i]]+= math.exp(pow(-1.0*distance,1.0/(Q-1.0)))/len(trainData)
    

    max_index, max_value = max(enumerate(classPosibility), key=operator.itemgetter(1))

    return max_index


def frnn(trainData, trainLabel, testData, testLabel, k):
    
    labTrainSet, labTestSet, legend = remapLabels(rlabTrainSet=trainLabel, rlabTestSet=testLabel)
    nClasses = len(legend)
    
    correct = 0
    index_testLabel = 0
    t1 = time()
    index=0
    for testDataInstance in testData:
        predict = clasifyClass(trainData,labTrainSet,testDataInstance,index,k,nClasses)
        progress(index_testLabel+1, len(testData))
        if legend[predict] == legend[labTestSet[index_testLabel]]:
            correct+=1
        index_testLabel+=1
        index+=1
        
    timeextract_feature= ("%0.5fs"%((time() - t1)/len(testData)))
    accouracy = round(((float(correct)/float(len(testData)))*100.0),2)
    return accouracy, round(((time() - t1)/len(testData)),5)