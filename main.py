import csv
import argparse
import textwrap
import sys
from time import time, strftime
from utils import average, best_accouracy, best_time, calculate_result, writer_report
from loader import load_data_stratified, load_data_unstratified
from dict import algorithm_list
from algoritma.knn import knn
from algoritma.adaknn import adaknn
from algoritma.fknn import fknn
from algoritma.it2fknn import it2fknn
from algoritma.frnn import frnn
from algoritma.frnn_frs import frnn_frs 
from algoritma.frnn_vqrs import frnn_vqrs

ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                             prog='NNP',
                             description=textwrap.dedent('''\
                               Nearest Neighborhood Package
                             -------------------------------------------
                               This is project for educational purpose
                               Author by : Ramadhan Adityo Kuncoro
                               4.8.2020

                             '''))

ap.add_argument("-f", "--file",
                help = "path to the file", metavar = "")
ap.add_argument("-kf", "--kfold", default=10, type=int,
                help = "k fold for crossvalidation (default: 10)", metavar = "")
ap.add_argument("-p", "--pca", action='store_true',
                help = "use pca (default: false)")
ap.add_argument("-n", "--n_comp", default=1, type=int,
                help = "n components for pca max is n coloumn (default: 1)", metavar = "")
ap.add_argument("-k", "--k", default=3, type=int, choices=range(3, 15, 2),
                help = "k for NN algorithm [3,5,7,9,11,13] (default: 3)", metavar = "")
ap.add_argument("-s", "--sampling", action='store_true',
                help = "sampling method use stratified sampling (default: false)")
ap.add_argument("-a", "--algorithm", default=[1], type=int, nargs='+',
                help = textwrap.dedent('''\
                                        You can choose more than one algorithm
                                        1  KNN ( default )
                                        2  AdaKNN
                                        3  FKNN
                                        4  It2FKNN
                                        5  FRNN
                                        6  FRNN FRS
                                        7  FRNN VQRS
                                        8  All
                                       '''),metavar = "")



args = vars(ap.parse_args())

filename = args["file"]
pca = args["pca"]
sampling = args["sampling"]
k = args["k"]
kfold = args["kfold"]
n_component = "-" if not pca else args["n_comp"]
algorithms = args["algorithm"]
if algorithms == [8]:
    algorithm_label =  ', '.join(x['label'] for x in algorithm_list)
    algorithms = [1,2,3,4,5,6,7]
else:
    algorithm_label =  ', '.join(algorithm_list[x-1]['label'] for x in algorithms)
    
if filename:
    if sampling:
        trainData, trainLabel, testData, testLabel=load_data_stratified(filename,kfold,pca,n_component)
    else:
        trainData, trainLabel, testData, testLabel=load_data_unstratified(filename,kfold,pca,n_component)
else:
    print("Please Use -f parameter and fill path to file data")
    sys.exit()


tanggal = strftime("%d%m%y-%H%M%S")
text_file = open("extract/report-k_"+str(k)+"-"+tanggal+".txt", "w")
text_file.write("==========================================\n")
text_file.write("Desc :\n")
text_file.write("\nfilename => %s \n" %filename)
text_file.write("pca => %s \n" %pca)
text_file.write("n_component => %s \n" %n_component)
text_file.write("nearest neighbor => %s \n" %k)
text_file.write("data per batch (train/test) => (%s/%s) \n" %(len(trainData[0]),len(testData[0])))
text_file.write("algorithm => %s \n" %algorithm_label)
text_file.write("==========================================\n")


for a in algorithms:
    index = a-1
    if a == 1 or a == 3 or a == 5 or a == 6 or a ==7:
        print "\n"+algorithm_list[index]['label']
        for x in  range(kfold):
            print "Batch " + str(x+1)
            
            if a == 1:
                t0=time()
                accouracy, timel = knn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
                timeload= ("%0.5f"%(time() - t0))
            elif a == 3:
                t0=time()
                accouracy, timel = fknn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
                timeload= ("%0.5f"%(time() - t0))
            elif a == 5:
                t0=time()
                accouracy, timel = frnn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
                timeload= ("%0.5f"%(time() - t0))
            elif a == 6:
                t0=time()
                accouracy, timel = frnn_frs(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
                timeload= ("%0.5f"%(time() - t0))
            elif a == 7:
                t0=time()
                accouracy, timel = frnn_vqrs(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
                timeload= ("%0.5f"%(time() - t0))
            print('')
            print ("result > ",accouracy,timel)
            print ("done in > ",timeload)
            algorithm_list[index]['data']['accouracy'].append(accouracy)
            algorithm_list[index]['data']['average_time'].append(timel)
            algorithm_list[index]['data']['total_time'].append(timeload)

        calculate_result(algorithm_list, index)            
        print "=========================================="
        print ("avg result  :%s, %s" %(algorithm_list[index]['average_result_accouracy'], algorithm_list[index]['average_result_time']))
        print ("best result :%s, %s" %(algorithm_list[index]['best_result_accouracy'], algorithm_list[index]['best_result_time']))
        print "=========================================="
        writer_report(text_file, algorithm_list, index, kfold)
        
    if a == 2:
        print "\n"+algorithm_list[index]['label']
        for x in  range(kfold):
            print "Batch " + str(x+1)
            t0=time()
            accouracy, timel, timeextract = adaknn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
            timeload= ("%0.5f"%(time() - t0))
            print('')
            print ("result > ",accouracy,timel, timeextract)
            print ("done in > ",timeload)
            algorithm_list[index]['data']['accouracy'].append(accouracy)
            algorithm_list[index]['data']['average_time'].append(timel)
            algorithm_list[index]['data']['total_time'].append(timeload)
            algorithm_list[index]['data']['radius_time'].append(timeextract)
            
        calculate_result(algorithm_list, index)            
        print "=========================================="
        print ("avg result  :%s, %s" %(algorithm_list[index]['average_result_accouracy'], algorithm_list[index]['average_result_time']))
        print ("best result :%s, %s" %(algorithm_list[index]['best_result_accouracy'], algorithm_list[index]['best_result_time']))
        print "=========================================="
        writer_report(text_file, algorithm_list, index, kfold)
            
    if a == 4:
        print "\n"+algorithm_list[index]['label']
        for x in  range(kfold):
            print "Batch " + str(x+1)
            t0=time()
            accouracy, timel, timeextract= it2fknn(trainData[x], trainLabel[x], testData[x], testLabel[x],k)
            timeload= ("%0.5f"%(time() - t0))
            print('')
            print ("result > ",accouracy,timel, timeextract)
            print ("done in > ",timeload)
            algorithm_list[index]['data']['accouracy'].append(accouracy)
            algorithm_list[index]['data']['average_time'].append(timel)
            algorithm_list[index]['data']['total_time'].append(timeload)
            algorithm_list[index]['data']['membership_time'].append(timeextract)
            
        calculate_result(algorithm_list, index)            
        print "=========================================="
        print ("avg result  :%s, %s" %(algorithm_list[index]['average_result_accouracy'], algorithm_list[index]['average_result_time']))
        print ("best result :%s, %s" %(algorithm_list[index]['best_result_accouracy'], algorithm_list[index]['best_result_time']))
        print "=========================================="
        writer_report(text_file, algorithm_list, index, kfold)

text_file.close()
