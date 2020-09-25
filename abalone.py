# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as linalg

get_ipython().run_line_magic('matplotlib', 'qt')

# Importing the dataset
dataset = pd.read_csv('abalone.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Basic summary statistics
print(dataset.describe())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Printing X_test
print(X_test)

# Printing X_train
print(X_train)

PCAAnalysis(X)

#PCA
def PCAAnalysis(X):
    L = len(Y[0])
    Ystr = X[1:,1:]
    print(str(Ystr))
    Y = np.zeros((len(Ystr),len(Ystr[1])),float)
    for i in range(len(Y)):
        for f in range(len(Y[i])):
            Y[i][f] = float(Ystr[i][f])
    Y.dtype = np.float
    Y = Y - np.ones((len(Y),1))*Y.mean(0)
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    rho = S/sum(S)
    rhoa = np.zeros((len(S),),float)
    for i in range(len(rho)):
        rhoa[i] = sum(rho[:i])
    
    K = 7;
    plt.figure()
    plt.plot(rhoa,'o-')
    plt.plot(rho,'o-')
    #plt.legend(["","Variation explained by the given principle component"])
    plt.show()
    
    Shat = np.vstack((np.diag(S[0:K]),np.zeros((L-K,K),float)))
    Vhat = V[0:K,:]
    #PC = np.dot(D,V[:,0:2])
    Xhat = np.dot(U,Shat)
    plt.figure()
    plt.plot(Xhat[:,0],Xhat[:,1],'o')
    plt.xlabel("PCA #1")
    plt.ylabel("PCA #2")       
    plt.show()
    
    plt.figure()
    plt.plot(Xhat[:,2],Xhat[:,3],'o')
    plt.xlabel("PCA #3")
    plt.ylabel("PCA #4")       
    plt.show()
    
    rings = np.zeros((len(y)-1),float)
    for i in range(1,len(y)):
        rings[i-1] = float(y[i])
    plt.figure()
    plt.plot(rings,Xhat[:,0],'o')
    plt.xlabel("Rings")
    plt.ylabel("PCA #1")       
    plt.show()
    
    
    plt.figure()
    for i in range(len(rings)):
        if X[i+1,0] == 'M':
            st = 'o'
        elif X[i+1,0] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i,0],Xhat[i,1],st,color = (rings[i]/30,0,1-rings[i]/30))
    plt.xlabel("PCA #1")
    plt.ylabel("PCA #2")       
    plt.show()
    
    
    plt.figure()
    for i in range(len(rings)):
        if X[i+1,0] == 'M':
            st = 'o'
        elif X[i+1,0] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i,2],Xhat[i,3],st,color = (rings[i]/30,0,1-rings[i]/30))
    plt.xlabel("PCA #3")
    plt.ylabel("PCA #4")       
    plt.show()
