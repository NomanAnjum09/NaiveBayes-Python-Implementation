import numpy as np
import math
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

class GussianNaiveBayes:
    def __init__(self):
        self.class_properties = []
        self.mean = []
        self.std_dev = []
    def train(self,X,Y):
        #####Find Prior Probablities of Classes
        classes , count  = np.unique(Y, return_counts=True)
        self.mean = np.zeros(shape=(len(classes),X.shape[1]))
        self.std_dev = np.zeros(shape=(len(classes),X.shape[1]))

        for i in range(len(classes)):
            self.class_properties.append({
                'class':classes[i],
                'count' : count[i],
                'prior_probabilty' : count[i]/np.sum(count)})
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                self.mean[Y[j],i]+=X[i][j]
        for i in range(len(classes)):
            self.mean[i]/=count[i]
        
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                self.std_dev[Y[j],i]+=pow(self.mean[Y[j],i]-X[i][j],2)
        for i in range(len(classes)):
            self.std_dev[i]/=count[i]

        print("Mean Of Features With Respect to Each Class")
        print(self.mean)
        print("Variance of Features With Respect to Each Class")
        print(self.std_dev)
        print("\n\n")
    def predict(self,X):
        if self.class_properties == []:
            print("Model Not Trained Yet")
            return
        prediction = []
        for i in range(X.shape[0]):
            list_of_features = []
            for b in range(len(self.class_properties)):
                list_of_features.append(self.class_properties[b]['prior_probabilty'])
            for j in range(X.shape[1]):
                for k in range(len(self.class_properties)):
                    x=X[j][i]
                    meo = self.mean[k][j]
                    var = self.std_dev[k][j]
                    n1 = pow((x-meo),2)*-1
                    d1 = 2*var
                    ex = math.exp(n1/d1)
                    denom = math.sqrt(2*math.pi*var)
                    list_of_features[k] *=  (ex/denom)
                   
            prediction.append(self.class_properties[list_of_features.index(max(list_of_features))]['class'])
        print(list_of_features)
        return prediction
        
data = pd.read_excel('parktraining.xlsx',index_col=None, header=None)
gnb = GussianNaiveBayes()
X_train = data.loc[:,:data.shape[1]-2]
Y_train = data.loc[:,data.shape[1]-1]
gnb.train(X_train,Y_train)
data = pd.read_excel('parktesting.xlsx',index_col=None, header=None)
X_test = data.loc[:,:data.shape[1]-2]
Y_test = data.loc[:,data.shape[1]-1]
ans = gnb.predict(X_test)
counter=0
print("Predcition\tActual Answer")
for i in range(len(ans)):
    print("{}\t\t{}".format(ans[i],Y_test[i]))
    if(Y_test[i]!=ans[i]):
        counter+=1

print("Error= {} out of {}".format(counter,len(ans)))
print("Accuracy = {}%".format(round((len(ans)-counter)*100/len(ans),2)))
print("\nConfusion Matrix")
print(confusion_matrix(Y_test,ans))



print("\n\nWith [0-1] Normalization Feature Scaling")
min_max_scaler = preprocessing.MinMaxScaler()
scaled_x_train = min_max_scaler.fit_transform(X_train)
scaled_x_test = min_max_scaler.fit_transform(X_test)
scaled_x_train= pd.DataFrame(scaled_x_train)
scaled_x_test= pd.DataFrame(scaled_x_test)
gnb1 = GussianNaiveBayes()
gnb1.train(scaled_x_train,Y_train)
ans = gnb1.predict(scaled_x_test)
counter=0
print("Predcition\tActual Answer")
for i in range(len(ans)):
    print("{}\t\t{}".format(ans[i],Y_test[i]))
    if(Y_test[i]!=ans[i]):
        counter+=1

print("Error= {} out of {}".format(counter,len(ans)))
print("Accuracy = {}%".format(round((len(ans)-counter)*100/len(ans),2)))
print("\nConfusion Matrix")
print(confusion_matrix(Y_test,ans))
