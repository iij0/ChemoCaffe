from net_spec import layers as L, params as P, to_proto
import net_spec
from caffe.proto import caffe_pb2
import caffe
from  sklearn.metrics import matthews_corrcoef,auc,roc_auc_score,recall_score,precision_score,f1_score,accuracy_score
import csv
import numpy as np
import time

def MakeNetwork(db, batch_size, layers, numClasses, deploy, act, dropout):

	#Create Data layer
	data, label = L.HDF5Data(source=db,batch_size=batch_size,ntop=2)

	#Add hidden layers
	top = data
	test = 0
	for x in range(0,len(layers)):
		top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'))
		print net_spec.params
		if(act == 1):
			top = L.ReLU(top,in_place=True)
		elif(act == 2):
			top = L.Sigmoid(top, in_place=True)
		elif(act == 3):
			top = L.TanH(top, in_place=True)
		else:
			print "Error, invalid activation function choice "
		if(dropout==True):
			top = L.Dropout(top, in_place=True, dropout_ratio = 0.5)

	#Add Output Layers
	output = L.InnerProduct(top, num_output=numClasses,weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'))
	if(deploy == False):
		loss = L.SoftmaxWithLoss(output,label)
		return to_proto(loss)	
	else:
		prob = L.Softmax(output)
		return to_proto(prob)

def WritePrototxt(test_size, layers, numClasses, act, dropout,folder):
	with open(folder+'train.prototxt','w') as f:
		print >>f, MakeNetwork('train.txt',500,layers,numClasses, False, act, dropout)
	with open(folder+'test.prototxt','w') as f:
		print >>f, MakeNetwork('test.txt',512,layers,numClasses, False, act, False)
	with open(folder+'deploy.prototxt','w') as f:
		print >>f, MakeNetwork('test.txt',test_size,layers,numClasses,True, act, False)	

def testConfig(numFolds,iters,folder):
	
	caffe.set_mode_gpu()

	MODEL_FILE = folder + 'deploy.prototxt'
	PRETRAINED = folder + 'KDD__iter_' + str(iters) + '.caffemodel'
	acc=0
	mcc=0
	RAUC=0
	Recall=0
	Precision=0
	F1_score = 0
	for x in range(0,numFolds):
		print "Writing txt files"
		with open(folder+'train.txt', 'w') as f:
			f.write('train_' + str(x+1) + '.h5' + '\n')
		with open(folder+'test.txt', 'w') as f:
			f.write('test_' + str(x+1) + '.h5' + '\n')
		print "Done"

		#Get Solver
		solver = caffe.SGDSolver(folder+'solver.prototxt')
		
		#Solve for given iterations
		solver.step(iters)

		#Load network
		net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
		
		#Forward data
		net.forward()
		
		#Record results
		results = net.blobs['Softmax1'].data

		#Read test labels
		test_size = 0
		y_test=[]
		reader=csv.reader(open(folder + 'test_'+str(x+1)+'.csv',"rU"),delimiter=',')
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
		acc+=accuracy_score(y_test, y_pred)
		mcc+=matthews_corrcoef(y_test,y_pred)
		RAUC+=roc_auc_score(y_test,y_pred_scores)
		Recall+=recall_score(y_test, y_pred,pos_label=1)
		Precision+=precision_score(y_test, y_pred,pos_label=1)
		F1_score+=f1_score(y_test, y_pred,pos_label=1)
	
	#Return average scores
	results = [acc/numFolds,mcc/numFolds,RAUC/numFolds,Recall/numFolds,Precision/numFolds,F1_score/numFolds]
	return results

t1=time.time()

res = []

#No Sigmoid

WritePrototxt(36163,[5,10],2,3,False,'blah/')
res.append(testConfig(1, 1000,'blah/'))

t2=time.time()

print res
print "Time Elapsed: ",t2-t1

res = np.asarray(res)
np.savetxt('results2.csv',res,fmt='%.10f',delimiter=',')

writer = csv.wrtier(open(results.csv,'wb'),delimiter=",")
writer.writerow("Acc,MCC,RAUC,Recall,Precision,F1")
for row in len(res):
	writer.writerow("DNN_"+str(row)+","+res[row])
