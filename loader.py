import csv
from time import time, strftime
from sklearn.model_selection import StratifiedKFold

def chunk_data(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append([int(last),int((last + avg)-1)])
    last += avg

  return out

def load_data_unstratified(filename, fold, ispca, n_component):
    tanggal = strftime("%d%m%y-%H%M%S")
    text_file = open("extract/data-"+tanggal+".txt", "w")
    t0 = time()
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)  
    trainingSetFold = []
    trainingSetTFold = []
    testSetDFold = []
    testSetTFold = []
    
    index = chunk_data(dataset,fold)
    
    
    for f in range(fold):
        trainingSet=[]
        testSet=[]
        for x in range(len(dataset)):            
            for y in range(len(dataset[0])-1):
                dataset[x][y] = float(dataset[x][y])
               
            if x >= index[f][0] and x <= index[f][1]:
                testSet.append(dataset[x])    
            else:
                trainingSet.append(dataset[x])
     
        if not ispca:
            trainingSetT = [row[-1] for row in trainingSet]
            trainingSetD = [row[0:len(row)-1] for row in trainingSet]
    
            testSetT = [row[-1] for row in testSet]
            testSetD = [row[0:len(row)-1] for row in testSet]  
        if ispca:
            from sklearn.decomposition import PCA
            trainingSetT = [row[-1] for row in trainingSet]
            trainingSetD = [row[0:len(row)-1] for row in trainingSet]
            
            testSetT = [row[-1] for row in testSet]
            testSetD = [row[0:len(row)-1] for row in testSet]
            t0 = time()
            pca = PCA(n_components=n_component)
            
            trainingSet = pca.fit_transform(trainingSetD)
            testSet = pca.transform(testSetD)
            timepreprocesss= ("%0.3fs"%(time() - t0))
            print("PCA from "+ str(len(dataset[0])-1) +" to "+str(n_component)+" done in %s" % timepreprocesss)
            text_file.write("PCA from %s to %s done in %s\n" % (str(len(dataset[0])-1),str(n_component),timepreprocesss))
            trainingSet=trainingSet.tolist()
            trainingSetD = trainingSet
            testSet=testSet.tolist()
            testSetD = testSet
    
        trainingSetFold.append(trainingSetD)
        trainingSetTFold.append(trainingSetT)
        testSetDFold.append(testSetD)
        testSetTFold.append(testSetT)
    timeload= ("%0.5fs"%(time() - t0))
    print "Load time > " +timeload + ", Dimension > "+ str(len(dataset)) +"*"+str(len(dataset[0]))
    text_file.write("Load time > %s ---- Dimension > %s * %s\n" %(timeload, str(len(dataset)), str(len(dataset[0]))))
    return trainingSetFold, trainingSetTFold, testSetDFold, testSetTFold
  
def load_data_stratified(filename, fold, ispca, n_component):
    tanggal = strftime("%d%m%y-%H%M%S")
    text_file = open("extract/data-"+tanggal+".txt", "w")
    t0 = time()
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)  
    trainingSetFold = []
    trainingSetTFold = []
    testSetDFold = []
    testSetTFold = []
    
    trainSet = []
    trainLabel = []
        
    for x in range(len(dataset)):
        for y in range(len(dataset[0])-1):
            dataset[x][y] = float(dataset[x][y])
        trainSet.append(dataset[x][:-1])
        trainLabel.append(dataset[x][-1])
    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(trainSet, trainLabel)
    
    
    for train_index, test_index in skf.split(trainSet, trainLabel):
        trainingSetD = []
        trainingSetT = []
        testSetT = []
        testSetD = []
        for y in train_index:
            trainingSetD.append(trainSet[y])
            trainingSetT.append(trainLabel[y])
        for y in test_index:
            testSetD.append(trainSet[y])
            testSetT.append(trainLabel[y])
        if ispca:
            from sklearn.decomposition import PCA
            t0 = time()
            pca = PCA(n_components=n_component)    
            trainingSet = pca.fit_transform(trainingSetD)
            testSet = pca.transform(testSetD)
            timepreprocesss= ("%0.3fs"%(time() - t0))
            print("PCA from "+ str(len(dataset[0])-1) +" to "+str(n_component)+" done in %s" % timepreprocesss)
            text_file.write("PCA from %s to %s done in %s\n" % (str(len(dataset[0])-1),str(n_component),timepreprocesss))
            trainingSet=trainingSet.tolist()
            trainingSetD = trainingSet
            testSet=testSet.tolist()
            testSetD = testSet
        trainingSetFold.append(trainingSetD)
        trainingSetTFold.append(trainingSetT)
        testSetDFold.append(testSetD)
        testSetTFold.append(testSetT)
    timeload= ("%0.5fs"%(time() - t0))
    print "Load time > " +timeload + ", Dimension > "+ str(len(dataset)) +"*"+str(len(dataset[0]))
    text_file.write("Load time > %s ---- Dimension > %s * %s\n" %(timeload, str(len(dataset)), str(len(dataset[0]))))
    return trainingSetFold, trainingSetTFold, testSetDFold, testSetTFold