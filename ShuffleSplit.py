import numpy as np
import csv
import h5py
from sklearn import cross_validation
from sys import argv
import os

Path=""

def readData(ifile):
    reader=csv.reader(open(ifile,'rb'),delimiter=',',quotechar='"')
    
    headers=reader.next()
    #print headers
    
    Labels=[]
    Features=[]
    count=0
    countA=0
    countIn=0
    for row in reader:
        count+=1
        if row[headers.index('Activity')]=='1':
            countA+=1
        else:
            countIn+=1
        Labels.append(row[headers.index('Activity')])
        Features.append(row[headers.index('ECFP4_1'):])
        
    
    
    print "Total number of instances: ",count
    print "Number of actives: ",countA
    print "Number of inactives: ",countIn
    
    return Labels,Features


def ShuffleSplit(Labels,Features,nfolds,tst_spl):
    rs = cross_validation.ShuffleSplit(len(Labels), n_iter=nfolds,test_size=float(tst_spl), random_state=69)
    count=0
    os.chdir(Path)
    print "Train-Test split %",100*float(1-tst_spl),100*float(tst_spl)
    print "Train-Test splits directory"
    print Path
    for train_index, test_index in rs:
        
        count+=1
        print count
        Train="train_"+str(count)+".csv"
        Test="test_"+str(count)+".csv"
        writer1=csv.writer(open(Train,'wb'),delimiter=',',quotechar='"')
        writer2=csv.writer(open(Test,'wb'),delimiter=',',quotechar='"')
        HDF5Train="train_"+str(count)+".h5"
        HDF5Test="test_"+str(count)+".h5"
        print "Fold: ",count
        tr_labels=[]
        tr_features=[]
        for j in train_index:
            tr_labels.append(Labels[j])
            tr_features.append(Features[j])
        tst_labels=[]
        tst_features=[]
        for k in test_index:
            tst_labels.append(Labels[k])
            tst_features.append(Features[k])
        print len(tr_labels)
        print len(tst_labels)
        print "Now writing train fold: ",count
        for j in range(len(tr_labels)):
            writer1.writerow([tr_labels[j]]+tr_features[j])
            
        for j in range(len(tst_labels)):
            writer2.writerow([tst_labels[j]]+tst_features[j])
        print "Now writing h5 files"
        
        Train_labels=np.array(tr_labels,dtype=float)
        Train_features=np.array(tr_features,dtype=float)
        Test_labels=np.array(tst_labels,dtype=float)
        Test_features=np.array(tst_features,dtype=float)
        
        with h5py.File(HDF5Train, 'w') as f:
            f['HDF5Data2'] =Train_labels.astype(np.float32)
            f['HDF5Data1'] =Train_features.astype(np.float32)
        with h5py.File(HDF5Test, 'w') as g:
            g['HDF5Data2'] =Test_labels.astype(np.float32)
            g['HDF5Data1'] =Test_features.astype(np.float32)

    
    return


if __name__ == '__main__':
	if len(argv) != 4:
		print "Usage: python ShuffeSplit.py /path/to/data.csv [number of splits] [fraction of data to be withheld for testing]\n"
	else:
		Labels,Features=readData(argv[1])
		ShuffleSplit(Labels,Features,argv[2],argv[3])
    
