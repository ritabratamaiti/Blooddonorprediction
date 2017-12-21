import pip
import numpy as numpy		
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import csv

X = []
Y = []
Z = []

def get_data(filename):
	x = []
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		for row in csvFileReader:
			x.append(int(row[0]))
			x.append(int(row[1]))
			x.append(int(row[2]))
			x.append(int(row[3]))
			X.append(x)
			x = []
			Y.append(int(row[4]))
	return

def ana_data2(filename):
	x = []
	ctr = 0.0
	ctr1 = ctr2 = ctr3 = ctr4 = 0.0	
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		for row in csvFileReader:
			ctr += 1
			x.append(int(row[0]))
			x.append(int(row[1]))
			x.append(int(row[2]))
			x.append(int(row[3]))
			prediction = clf.predict(x)
			pred_svm = clf_svm.predict(x)
			pred_per = clf_perceptron.predict(x)
			pred_KNN = clf_KNN.predict(x)
			if(prediction[0]==int(row[4])): 
				ctr1 += 1
			if(pred_svm[0]==int(row[4])): 
				ctr2 += 1
			if(pred_per[0]==int(row[4])): 
				ctr3 += 1
			if(pred_KNN[0]==int(row[4])): 
				ctr4 += 1
			x = []
	print("\n",ctr,ctr1,ctr2,ctr3,ctr4)
	print("\n",ctr1/ctr,ctr2/ctr,ctr3/ctr,ctr4/ctr)
	fh = open("out.txt", "a") 
	fh.writelines(["\n",str(ctr1/ctr),str(ctr2/ctr),str(ctr3/ctr),str(ctr4/ctr)]) 
	fh.close  
	return 

get_data("donor1.csv")

clf = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()


clf = clf.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

ana_data2("donor2.csv")
 
