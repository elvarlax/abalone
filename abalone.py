# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as linalg
import categoric2numeric as c2n
from IPython import get_ipython

# get_ipython().run_line_magic('matplotlib', 'qt')

# Importing the dataset
dataset = pd.read_csv('abalone.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Printing X_train
print(X_train)

Y = np.zeros((len(X),len(X[1])-1),float)
for i in range(len(Y)):
    for f in range(1,len(Y[i])):
        Y[i][f-1] = float(X[i][f])
        
age = np.zeros(len(y),float)
for i in range(len(y)):
    age[i] = float(y[i])+1.5
    
MFIstr = X[:,0]
MFI,b = c2n.categoric2numeric(X[:,0])

X = np.hstack((MFI,Y))

#PCA
def PCAAnalysis(Y,y,MFI):
    L = len(Y[0])
    Y.dtype = np.float
    Y = Y - np.ones((len(Y), 1)) * Y.mean(0)
    U, S, V = linalg.svd(Y, full_matrices=False)

    rho = S / sum(S)
    rhoa = np.zeros((len(S),), float)
    for i in range(len(rho)):
        rhoa[i] = sum(rho[:i])

    K = 7;
    plt.figure()
    plt.plot(rhoa, 'o-')
    plt.plot(rho, 'o-')
    # plt.legend(["","Variation explained by the given principle component"])
    plt.show()

    Shat = np.vstack((np.diag(S[0:K]), np.zeros((L - K, K), float)))
    Vhat = V[0:K, :]
    # PC = np.dot(D,V[:,0:2])
    Xhat = np.dot(U, Shat)
    plt.figure()
    plt.plot(Xhat[:, 0], Xhat[:, 1], 'o')
    plt.xlabel("PCA #1")
    plt.ylabel("PCA #2")
    plt.show()

    plt.figure()
    plt.plot(Xhat[:, 2], Xhat[:, 3], 'o')
    plt.xlabel("PCA #3")
    plt.ylabel("PCA #4")
    plt.show()
    
    plt.figure()
    plt.plot(age,Xhat[:,0],'o')
    plt.xlabel("Rings")
    plt.ylabel("PCA #1")
    plt.show()

    plt.figure()
    for i in range(len(age)):
        if MFI[i] == 'M':
            st = 'o'
        elif MFI[i] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i,0],Xhat[i,1],st,color = (age[i]/33,0,1-age[i]/33))
    plt.xlabel("PCA #1")
    plt.ylabel("PCA #2")
    plt.show()

    plt.figure()
    for i in range(len(age)):
        if MFI[i] == 'M':
            st = 'o'
        elif MFI[i] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i,2],Xhat[i,3],st,color = (age[i]/33,0,1-age[i]/33))
    plt.xlabel("PCA #3")
    plt.ylabel("PCA #4")
    plt.show()
    
    return V

def boxplot(dataset):
    attributeNames = dataset.columns
    data = dataset.iloc[:, 1:-1]
    plt.boxplot(data)
    plt.xticks(range(1, 6), attributeNames)
    plt.ylabel('cm')
    plt.title('Boxplot')
    plt.show()


# PCAAnalysis(X)

boxplot(dataset)
