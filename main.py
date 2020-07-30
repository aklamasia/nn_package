import csv
import argparse
import random
import math
import operator
from time import time, strftime
from sklearn.decomposition import PCA
from crossvalidation2 import loadDataset
from algoritma.knn import knn
from algoritma.adaknn import adaknn
from algoritma.fknn import fknn
from algoritma.it2fknn import it2fknn
from algoritma.frnn import frnn
from algoritma.frnn_frs import frnn_frs 
from algoritma.frnn_vqrs import frnn_vqrs
from algoritma.if_knn import if_knn

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
text_file.write("==========================================\n")
text_file.write("Desc :\n")
text_file.write("\nfilename => %s \n" %filename)
text_file.write("pca => %s \n" %pca)
text_file.write("n_component => %s \n" %n_component)
text_file.write("nearest neighbor => %s \n" %k)
text_file.write("==========================================\n")

# list_knn=[]
# print "\nKNN"
# text_file.write("\nKNN\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel = knn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_knn.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_knn)/10, sum(row[1] for row in list_knn)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_knn)/10, sum(row[1] for row in list_knn)/10))))
# print (("best result :", max(row[0] for row in list_knn), min(row[1] for row in list_knn)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_knn), min(row[1] for row in list_knn)))))
# print "=========================================="
# text_file.write("==========================================\n")

# list_adaknn=[]
# print "\nAdaptive KNN"
# text_file.write("\nAdaptive KNN\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel, timeextract = adaknn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print "assign radius > " + timeextract
#     text_file.write("\nassign radius > %s\n" %timeextract)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_adaknn.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_adaknn)/10, sum(row[1] for row in list_adaknn)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_adaknn)/10, sum(row[1] for row in list_adaknn)/10))))
# print (("best result :", max(row[0] for row in list_adaknn), min(row[1] for row in list_adaknn)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_adaknn), min(row[1] for row in list_adaknn)))))
# print "=========================================="
# text_file.write("==========================================\n")

list_fknn=[]
print "\nFKNN"
text_file.write("\nFKNN\n")
for x in  range(10):
    t0=time()
    print "Batch " + str(x+1)
    text_file.write("Batch " + str(x+1))
    accouracy, timel = fknn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
    print('')
    print ("result > ",accouracy,timel)
    text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
    timeload= ("%0.5f"%(time() - t0))
    print ("done in > ",timeload)
    text_file.write(" done in > %s\n" %(timeload))
    list_fknn.append((accouracy,timel))
print "=========================================="
text_file.write("==========================================\n")
print (("avg result  :", sum(row[0] for row in list_fknn)/10, sum(row[1] for row in list_fknn)/10))
text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_fknn)/10, sum(row[1] for row in list_fknn)/10))))
print (("best result :", max(row[0] for row in list_fknn), min(row[1] for row in list_fknn)))
text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_fknn), min(row[1] for row in list_fknn)))))
print "=========================================="
text_file.write("==========================================\n")
    
# list_it2fknn=[]
# print "\nIT2FKNN"
# text_file.write("\nIT2FKNN\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel, timeextract= it2fknn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print "assign membership > " + timeextract
#     text_file.write("\nassign membership > %s\n" %timeextract)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_it2fknn.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_it2fknn)/10, sum(row[1] for row in list_it2fknn)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_it2fknn)/10, sum(row[1] for row in list_it2fknn)/10))))
# print (("best result :", max(row[0] for row in list_it2fknn), min(row[1] for row in list_it2fknn)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_it2fknn), min(row[1] for row in list_it2fknn)))))
# print "=========================================="
# text_file.write("==========================================\n")

# list_frnn=[]
# print "\nFRNN"
# text_file.write("\nFRNN\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel = frnn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_frnn.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_frnn)/10, sum(row[1] for row in list_frnn)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_frnn)/10, sum(row[1] for row in list_frnn)/10))))
# print (("best result :", max(row[0] for row in list_frnn), min(row[1] for row in list_frnn)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_frnn), min(row[1] for row in list_frnn)))))
# print "=========================================="
# text_file.write("==========================================\n")
    
# list_frnn_frs=[]
# print "\nFRNN-FRS"
# text_file.write("\nFRNN-FRS\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel = frnn_frs(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_frnn_frs.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_frnn_frs)/10, sum(row[1] for row in list_frnn_frs)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_frnn_frs)/10, sum(row[1] for row in list_frnn_frs)/10))))
# print (("best result :", max(row[0] for row in list_frnn_frs), min(row[1] for row in list_frnn_frs)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_frnn_frs), min(row[1] for row in list_frnn_frs)))))
# print "=========================================="
# text_file.write("==========================================\n")
#     
# list_frnn_vqrs=[]
# print "\nFRNN-VQRS"
# text_file.write("\nFRNN-VQRS\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel = frnn_vqrs(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_frnn_vqrs.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_frnn_vqrs)/10, sum(row[1] for row in list_frnn_vqrs)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_frnn_vqrs)/10, sum(row[1] for row in list_frnn_vqrs)/10))))
# print (("best result :", max(row[0] for row in list_frnn_vqrs), min(row[1] for row in list_frnn_vqrs)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_frnn_vqrs), min(row[1] for row in list_frnn_vqrs)))))
# print "=========================================="
# text_file.write("==========================================\n")

# list_if_knn=[]
# print "\nIF_KNN"
# text_file.write("\nIF_KNN\n")
# for x in  range(10):
#     t0=time()
#     print "Batch " + str(x+1)
#     text_file.write("Batch " + str(x+1))
#     accouracy, timel, timeextract= if_knn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
#     print "assign membership > " + timeextract
#     text_file.write("\nassign membership > %s\n" %timeextract)
#     print ("result > ",accouracy,timel)
#     text_file.write(" result > %s, %s  (%s / %s)\n" %(accouracy,timel,(len(trainData[x])),(len(testData[x]))))
#     timeload= ("%0.5f"%(time() - t0))
#     print ("done in > ",timeload)
#     text_file.write(" done in > %s\n" %(timeload))
#     list_if_knn.append((accouracy,timel))
# print "=========================================="
# text_file.write("==========================================\n")
# print (("avg result  :", sum(row[0] for row in list_if_knn)/10, sum(row[1] for row in list_if_knn)/10))
# text_file.write( (("avg result  : %s, %s\n" %(sum(row[0] for row in list_if_knn)/10, sum(row[1] for row in list_if_knn)/10))))
# print (("best result :", max(row[0] for row in list_if_knn), min(row[1] for row in list_if_knn)))
# text_file.write((("best result : %s, %s\n" % (max(row[0] for row in list_if_knn), min(row[1] for row in list_if_knn)))))
# print "=========================================="
# text_file.write("==========================================\n")

text_file.close()
