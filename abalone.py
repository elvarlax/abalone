# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.linalg as linalg
import categoric2numeric as c2n
from IPython import get_ipython
import similarity as sim


# get_ipython().run_line_magic('matplotlib', 'qt')

def pca(Y, y, MFI):
    L = len(Y[0])
    Y.dtype = np.float
    Y = Y - np.ones((len(Y), 1)) * Y.mean(0)
    U, S, V = linalg.svd(Y, full_matrices=False)

    rho = S / sum(S)
    rhoa = np.zeros((len(S),), float)
    for i in range(len(rho)):
        rhoa[i] = sum(rho[:i])

    K = 7
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
    plt.plot(age, Xhat[:, 0], 'o')
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
        plt.plot(Xhat[i, 0], Xhat[i, 1], st, color=(age[i] / 33, 0, 1 - age[i] / 33))
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
        plt.plot(Xhat[i, 2], Xhat[i, 3], st, color=(age[i] / 33, 0, 1 - age[i] / 33))
    plt.xlabel("PCA #3")
    plt.ylabel("PCA #4")
    plt.show()

    return V


def box_plot(x_val, y_val, data):
    plt.figure()
    ax = sns.boxplot(x=x_val, y=y_val, data=data, palette="Set1")
    plt.show()


def matrix_plot(data):
    sns.pairplot(data)
    plt.show()

def similarity_analysis(X,Y, attribute_names):
    method='cor'
    similar_mat= np.zeros((len(X), len(X)), float)
    for attribute in attribute_names:
        for attribute in attribute_names:
            sim(X, Y, method)

if __name__ == "__main__":
    # Import dataset
    dataset = pd.read_csv('abalone.csv') 
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    attributeNames = np.asarray(dataset.columns[:-1])
    # Create a age column from the Rings column + 1.5
    dataset['Age'] = dataset['Rings'] + 1.5
    dataset.drop('Rings', axis=1, inplace=True)

    # Move this to a function
    Y = np.zeros((len(X), len(X[1]) - 1), float)
    for i in range(len(Y)):
        for f in range(1, len(Y[i])):
            Y[i][f - 1] = float(X[i][f])

    age = np.zeros(len(y), float)
    for i in range(len(y)):
        age[i] = float(y[i]) + 1.5

    MFIstr = X[:, 0]
    MFI, b = c2n.categoric2numeric(X[:, 0])
    X = np.hstack((MFI, Y))

    # pca(Y, y, MFI)

    df = dataset[dataset.columns[1:-1]]
    box_plot(dataset['Sex'], dataset['Age'], df)
    matrix_plot(df)



    




