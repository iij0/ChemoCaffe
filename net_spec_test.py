"""
@Author: Keith Monaghan
@Date: 7/24/15
@Title: ChemoCaffe
@Description: Automates the testing of multiple network configurations in Caffe
"""

from net_spec import layers as L, params as P, to_proto
import net_spec
from caffe.proto import caffe_pb2
import caffe
from  sklearn.metrics import matthews_corrcoef,auc,roc_auc_score,recall_score,precision_score,f1_score,accuracy_score
import csv
import numpy as np
import time
import os

class CaffeNet:

	_test_size=0
	_batch_size=0
	_epochs=0
	_lr=0
	_test_interval=0
	_data_path= ""
	_solver=""
	_output=""
	_folds=0
	_train_size=0
	_test_size=0

	#Constructor 
	#Initializes variables which are usually constant across different network configurations
	def __init__(self,folds,epochs,test_interval,lr,batch_size,data,solver,output):
		self._folds=folds
		self._test_interval=test_interval
		self._data_path=data
		self._solver=solver
		self._output=output
		self._epochs=epochs
		self._lr=lr
		self._batch_size=batch_size
		self.getDataSize()
		writer=csv.writer(open(output,'a'),delimiter=',')
		writer.writerow(['Epochs','Layers','Activation','Dropout','L2','Acc','MCC','RAUC','Recall','Precision','F1'])

	#Helper Function for constructor
	#Reads size of test data set 
	#Converts epochs to iterations
	def getDataSize(self):
		reader1=csv.reader(open(self._data_path+'train_1.csv',"rU"))
		reader2=csv.reader(open(self._data_path+'test_1.csv',"rU"))
		self._train_size = 0
		self._test_size = 0
		for row in reader1:
			self._train_size = self._train_size+1
		for row in reader2:
			self._test_size = self._test_size+1

		self._epochs = (self._epochs*self._train_size)/self._batch_size
		self._test_interval = (self._test_interval*self._train_size)/self._batch_size

	#Builds the network with specified hyperparameters
	def MakeNetwork(self,db,batch_size,layers,numClasses,deploy,act,dropout,L2):
	
		#Create Data layer
		data, label = L.HDF5Data(source=db,batch_size=batch_size,ntop=2)
	
		#Add hidden layers
		top = data
		test = 0
		for x in range(0,len(layers)):
			if(L2):
				top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'),param=[dict(decay_mult=1)])
			else:
				top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'))
	
			if(act == 1):
				top = L.ReLU(top,in_place=True)
			elif(act == 2):
				top = L.Sigmoid(top, in_place=True)
			elif(act == 3):
				top = L.TanH(top, in_place=True)
			else:
				print "Error, invalid activation function choice "
			if(dropout!=0):
				top = L.Dropout(top, in_place=True, dropout_ratio = dropout)
	
		#Add Output Layers
		output = L.InnerProduct(top, num_output=numClasses,weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'))
		if(deploy == False):
			loss = L.SoftmaxWithLoss(output,label)
			return to_proto(loss)	
		else:
			prob = L.Softmax(output)
			return to_proto(prob)
	
	#Generates train, test, and deploy network definitions
	#Writes networks to .prototxt format to be read by Caffe
	def WritePrototxt(self,layers,numClasses,act,dropout,L2):
		with open('train.prototxt','w') as f:
			print >>f, self.MakeNetwork('train.txt',self._batch_size,layers,numClasses,False,act,dropout,L2)
		with open('test.prototxt','w') as f:
			print >>f, self.MakeNetwork('test.txt',self._batch_size,layers,numClasses,False,act,0,L2)
		with open('deploy.prototxt','w') as f:
			print >>f, self.MakeNetwork('test.txt',self._test_size,layers,numClasses,True,act,0,L2)	
	
	#Trains and evaluates the performance of a network across multiple folds
	def testConfig(self,layers,numClasses,act,dropout,L2):
		
		#Generate .prototxt files
		self.WritePrototxt(layers,numClasses,act,dropout,L2)
		#Enable GPU Mode
		caffe.set_mode_gpu()
		
		MODEL_FILE = 'deploy.prototxt'

		#Set iters & learning rate in solver file
		with open('solver_new.prototxt','w') as new_file:
			with open(self._solver,'r') as old_file:
				for line in old_file:
					if 'max_iter:' in line:
						new_file.write('max_iter:'+str(self._epochs)+"\n")
					elif 'snapshot:' in line:
						new_file.write('snapshot:'+str(self._test_interval)+"\n")
					elif 'base_lr:' in line:
						new_file.write('base_lr:'+str(self._lr)+"\n")
					else:
						new_file.write(line)
		os.rename('solver_new.prototxt',self._solver)

		temp = []

		for i in range(0,self._folds):

			#Write text files specifying the databse path
			print "Writing txt files"
			with open('train.txt', 'w') as f:
				f.write(self._data_path+'train_' + str(i+1) + '.h5' + '\n')
			with open('test.txt', 'w') as f:
				f.write(self._data_path+'test_' + str(i+1) + '.h5' + '\n')
			print "Done"

			#Get Solver
			solver = caffe.SGDSolver(self._solver)
			
			#Train network
			solver.solve()

			for x in range(0,int(self._epochs/self._test_interval)):
				PRETRAINED = 'KDD__iter_' + str((x+1)*self._test_interval) + '.caffemodel'
		
				#Load network
				net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
				
				#Forward data
				net.forward()
				
				#Record results
				results = net.blobs['Softmax1'].data
		
				#Read test labels
				test_size = 0
				y_test=[]
				reader=csv.reader(open(self._data_path + 'test_'+str(i+1)+'.csv',"rU"),delimiter=',')
				for row in reader:
					y_test.append(int(row[0]))
					test_size = test_size+1
				
				#Read predictions
				y_pred = []
				y_pred_scores = []
				for y in range(0,test_size):
					y_pred.append(int(results[y].argmax()))
					y_pred_scores.append(float(results[y][1][0][0]))
		
				#Score Predictions
				acc=accuracy_score(y_test, y_pred)
				mcc=matthews_corrcoef(y_test,y_pred)
				RAUC=roc_auc_score(y_test,y_pred_scores)
				Recall=recall_score(y_test, y_pred,pos_label=1)
				Precision=precision_score(y_test, y_pred,pos_label=1)
				F1_score=f1_score(y_test, y_pred,pos_label=1)

				#Store scores in temporary list
				temp.append([self._test_interval*(x+1),acc,mcc,RAUC,Recall,Precision,F1_score])
		
		#Open output writer	in append mode	
		writer=csv.writer(open(self._output,'a'),delimiter=',')
		
		#Average results for each test interval and print to output file
		for y in range(0,self._epochs/self._test_interval):
			epochs=0
			acc=0
			mcc=0
			RAUC=0
			Recall=0
			Precision=0
			F1_score=0
			for x in range(y,len(temp),self._epochs/self._test_interval):
				epochs=((temp[x][0]*self._batch_size)/self._train_size)
				acc+=temp[x][1]
				mcc+=temp[x][2]
				RAUC+=temp[x][3]
				Recall+=temp[x][4]
				Precision+=temp[x][5]
				F1_score+=temp[x][6]

			#Write output
			acts = ["ReLU","Sigmoid","TanH"]
			writer.writerow([epochs+1,str(layers),acts[act-1],str(dropout),str(L2),acc/self._folds,mcc/self._folds,RAUC/self._folds,Recall/self._folds,Precision/self._folds,F1_score/self._folds])

if __name__ == '__main__':
	t1=time.time()
	
	#Create instance of CaffeNet
	test = CaffeNet(2,3,1,0.01,512,'/users/kmonaghan/caffe/Automation/','/users/kmonaghan/caffe/Automation/solver.prototxt','/users/kmonaghan/caffe/Automation/results.csv')
	
	#Test the performance of a specific configuration
	test.testConfig([10,10],2,1,0.5,True)
	
	t2=time.time()
	print "Time Elapsed: ",t2-t1