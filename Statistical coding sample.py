##load packages#########################################
import numpy as np
from sklearn import preprocessing# import RobustScaler
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
########################################################
#define data input function
def getGender(file):
        print "Reading "+file+"..."
        f=open(file)
        gender=[]
        for line in f:
                line=line.split()
		gender.append(int(line[0]))
        f.close()
        return(gender)

def ReadData(file,start,end):
	print "Reading "+file+"..."
	myData=[]
	f=open(file)
	for line in f:
		line=line.split()
		tmp=[]
		if end==0: end=len(line)
		for i in range(start,end): tmp.append(float(line[i]))
		myData.append(tmp)
	f.close()
	myData=preprocessing.normalize(np.array(myData))
	print len(tmp)
	return(myData.tolist())

def combinePredictVariable(image_features,dis_train,images,words):
	PredictV=[]
	for i in range(len(images)):
		PredictV.append(image_features[i]+dis_train[i]+images[i]+words[i])
	return(PredictV)

def Accuracy(x,y):
	n=0.0
	for i in range(len(x)):
		if x[i]==y[i]: n+=1.0
	print n/len(x)

if __name__=='__main__':
	import random
	######import train data##########################################
	gender=getGender("../train/genders_train.txt")
	image_features=ReadData("../train/image_features_train.txt",0,0)
	images=ReadData("../test/train_30pca.txt",0,30)
	train_dis=ReadData("../train/distance2AI.txt",0,0)
	words=ReadData("../train/words_train.txt",0,0)
	PredictV=combinePredictVariable(image_features,train_dis,images,words)
	#################################################################
	#####build models################################################
	print "modeling..."
	RF_clf = RandomForestClassifier(n_estimators=100)
	svm_clf = svm.LinearSVC(C=10)
	knn_clf = KNeighborsClassifier()
	logit_clf = LogisticRegression(C=100, penalty='l1', tol=0.01)
	NB_clf = GaussianNB()
	eclf = EnsembleClassifier(clfs=[RF_clf,logit_clf,svm_clf],voting='hard',weights=[1,1,1])
	#bdt = AdaBoostClassifier(base_estimator=eclf,n_estimators=20)
	#####cross validation###########################################
	scores = cross_val_score(eclf, PredictV, gender)
	print scores.mean()
	################################################################
	
	#####fitting data##############################################
	model=eclf.fit(PredictV, gender)
	#####load test data############################################
	test_image_features=ReadData("../test/image_features_test.txt",0,0)
	test_dis=ReadData("../test/distance2AI.txt",0,0)
	test_images=ReadData("../test/test_30pca.txt",0,30)
	test_words=ReadData("../test/words_test.txt",0,0)
	testData=combinePredictVariable(test_image_features,test_dis,test_images,test_words)
	################################################################
	#####prediction using test data#################################
	testGender=model.predict(testData)
	################################################################
	#####output####################################################
	h=open("../test/submit_with_dis.RF1.LR1.LinearSVM1.txt",'w')
	Trans={0:'0',1:'1'}
	for i in range(len(testGender)):
		h.write(Trans[testGender[i]]+"\n")
	
	h.close()
	

