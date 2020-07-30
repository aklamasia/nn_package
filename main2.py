import csv
import argparse
import random
import math
import operator
from time import time, strftime
from sklearn.decomposition import PCA
from crossvalidation import loadDataset
from algoritma.knn import knn
from algoritma.fknn import fknn
from algoritma.it2fknn import it2fknn
from algoritma.frnn import frnn
from algoritma.frnn_frs import frnn_frs 
from algoritma.frnn_vqrs import frnn_vqrs
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help = "path to the file")
ap.add_argument("-p", "--pca", default="False", help = "use mode pca, default false")
ap.add_argument("-n", "--n_comp", default=100, help = "n_components for pca")
args = vars(ap.parse_args())
filename = args["file"]
pca = args["pca"]
n_component = int(args["n_comp"])

trainData, trainLabel, testData, testLabel=loadDataset(filename,10,pca,n_component)
k = 7

tanggal = strftime("%d%m%y-%H%M%S")
text_file = open("extract/report-k_"+str(k)+"-"+tanggal+".txt", "w")

list_svm=[]
print "\nSVM"
text_file.write("\nSVM\n")
avg_accouracy = 0
avg_time =0
for x in  range(10):
    t1 = time()
    result = OneVsOneClassifier(LinearSVC(random_state=0)).fit(trainData[x], trainLabel[x]).predict(testData[x])
    timep= ("%0.5f"%((time() - t1)/len(testData[x])))
    acc = 0
   
    for y in range(len(result)):
        if result[y]==testLabel[x][y]:
            acc +=1
    
    print "Batch " + str(x+1)
    text_file.write("Batch " + str(x+1))
    accouracy = (float(acc) / float(len(result)))*100
    print ("result > ",accouracy,timep)
    text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timep,(len(trainData[x])),(len(testData[x]))))
    timeload= ("%0.5f"%(time() - t1))
    print ("done in > ",timeload)
    text_file.write(" done in > %s\n" %(timeload))
    avg_accouracy+=accouracy
    list_svm.append((accouracy,float(timep)))

print "=========================================="
text_file.write("==========================================\n")
print (("avg result  :", sum(row[0] for row in list_svm)/10, sum(row[1] for row in list_svm)/10))
text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_svm)/10, sum(row[1] for row in list_svm)/10))))
print (("best result :", max(row[0] for row in list_svm), min(row[1] for row in list_svm)))
text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_svm), min(row[1] for row in list_svm)))))
print "=========================================="
text_file.write("==========================================\n")



text_file.close()
