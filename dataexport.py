import csv
import argparse
import random
import math
import operator
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help = "path to the file")
ap.add_argument("-n", "--n_comp", default=100, help = "n_components for pca")
args = vars(ap.parse_args())
filename = args["file"]
n_component = int(args["n_comp"])

with open(filename, 'rb') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)

trainSet = []
trainLabel = []
        
for x in range(len(dataset)):
    for y in range(len(dataset[0])-1):
        dataset[x][y] = float(dataset[x][y])
    trainSet.append(dataset[x][:-1])
    trainLabel.append(dataset[x][-1])
    
pca = PCA(n_components=n_component)    
trainingSet = pca.fit_transform(trainSet)
trainingSet=trainingSet.tolist()

for x in range(len(trainingSet)):
    trainingSet[x].append(trainLabel[x])
    
with open("pca-" + str(n_component)+ ".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(trainingSet)