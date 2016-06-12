#!/bin/python



from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import os
from sys import argv



def ECFP(ifile,ofile,diam,bits):
    
    reader=csv.reader(open(ifile,'rb'),delimiter=',',quotechar='"')
    writer=csv.writer(open(ofile,'wb'),delimiter=',',quotechar='"')
    count=0
    headers=reader.next()
    print headers
    ECFP=[]
    for j in range(1024):
	ECFP.append("ECFP4_"+str(j+1))
    writer.writerow(headers+ECFP)
    for row in reader:
        count+=1        
	if count%10000==0:
            print count
        mol=Chem.MolFromSmiles(str(row[0]))
        fp=AllChem.GetMorganFingerprintAsBitVect(mol,int(diam),nBits=int(bits))
        temp=[]
        for j in range(len(fp)):
            temp.append(fp[j])
        writer.writerow(row[0:3]+temp)
        
    return



if __name__ == '__main__':
    ECFP(argv[1],
         "output.csv",2,1024)
