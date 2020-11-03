import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.linalg as linalg
import similarity as sim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def pca(Y, y, MFI):
    L = len(Y[0])
    Y.dtype = np.float

    Y = Y - np.ones((len(Y), 1)) * Y.mean(0)  # Translate the data to around origin
    Y = Y / (np.ones((len(Y), 1)) * Y.std(0))
    U, S, V = linalg.svd(Y, full_matrices=False)  #

    rho = S / sum(S)
    rhoa = np.zeros((len(S),), float)
    for i in range(len(rho)):
        rhoa[i] = sum(rho[:(i + 1)])
        rhoa[i] = sum(rho[:(i + 1)])

    K = 4
    plt.figure()
    plt.plot(rhoa, 'o-')
    plt.plot(rho, 'o-')
    # plt.legend(["","Variation explained by the given principle component"])
    plt.show()

    Shat = np.vstack((np.diag(S[0:K]), np.zeros((L - K, K), float)))
    Vhat = V[0:K, :]
    # Print for the directions
    # print(str(Vhat))
    Xhat = np.dot(U, Shat)
    plt.figure()
    plt.plot(Xhat[:, 0], Xhat[:, 1], 'o')
    plt.xlabel("PC #1")
    plt.ylabel("PC #2")
    plt.show()

    plt.figure()
    plt.plot(Xhat[:, 2], Xhat[:, 3], 'o')
    plt.xlabel("PC #3")
    plt.ylabel("PC #4")
    plt.show()

    plt.figure()
    plt.plot(age, Xhat[:, 0], 'o')
    plt.xlabel("Age")
    plt.ylabel("PC #1")
    plt.show()

    plt.figure()
    for i in range(len(age)):
        if MFI[i] == 'M':
            st = 'o'
        elif MFI[i] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i, 0],
                 Xhat[i, 1],
                 st,
                 color=(age[i] / 33, 0, 1 - age[i] / 33))
    plt.xlabel("PC #1")
    plt.ylabel("PC #2")
    plt.show()

    plt.figure()
    for i in range(len(age)):
        if MFI[i] == 'M':
            st = 'o'
        elif MFI[i] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i, 2],
                 Xhat[i, 3],
                 st,
                 color=(min(age[i] / 15, 1), 0, max(1 - age[i] / 15, 0)))
    plt.xlabel("PC #3")
    plt.ylabel("PC #4")
    plt.show()

    plt.figure()
    for i in range(len(age)):
        if MFI[i] == 'M':
            st = 'o'
        elif MFI[i] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i, 0],
                 Xhat[i, 3],
                 st,
                 color=(min(age[i] / 15, 1), 0, max(1 - age[i] / 15, 0)))
    plt.xlabel("PC #1")
    plt.ylabel("PC #4")
    plt.show()

    v = np.vstack((MFI, Xhat[:, :K].T))
    df2 = pd.DataFrame(v.T,
                       columns=['Sex', 'PC1', 'PC2', 'PC3', 'PC4'])
    plt.figure()
    sns.pairplot(df2, hue='Sex', palette="Set1", markers=["s", "o", "D"])
    plt.show()

    return V


def box_plot(x_val, y_val, data):
    plt.figure()
    ax = sns.boxplot(x=x_val, y=y_val, data=data, palette="Set1")
    plt.show()


def matrix_plot(data):
    plt.figure()
    ax = sns.pairplot(data, hue='Sex', palette="Set1", markers=["s", "o", "D"])
    plt.show()


def similarity_analysis(X, M, method='cor'):
    sim_mat = np.zeros((M, M), float)
    for j in range(M):
        for i in range(M):
            # If it does not work you need to comment 
            # "keepdims=True" argument in the mean() function in stats.py
            sim_mat[i, j] = sim.similarity(X.iloc[:, i], X.iloc[:, j], method)
    return sim_mat


def heatmap_plot(corr_data):
    plt.figure()
    # Masking unique values
    mask = np.zeros_like(corr_data)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_data,
                mask=mask,
                xticklabels=attributeNames,
                yticklabels=attributeNames,
                square=True,
                cbar_kws={'label': 'Correlation'})
    plt.xlabel('Attribute name')
    plt.ylabel('Attribute name')
    plt.show()


def correlation_plot(d):
    nrows, ncols = int(len(d) / 2), int(len(d) / 2)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            key = list(d.keys())[k]
            corr_data = d[key]
            mask = np.zeros_like(corr_data)
            mask[np.triu_indices_from(mask)] = True
            if k == 0 or k == 2:
                sns.heatmap(corr_data,
                            mask=mask,
                            xticklabels=attributeNames,
                            yticklabels=attributeNames,
                            square=True,
                            ax=ax[i, j],
                            vmin=0,
                            vmax=1,
                            cmap="coolwarm",
                            annot=True)
            else:
                sns.heatmap(corr_data,
                            mask=mask,
                            xticklabels=attributeNames,
                            yticklabels=attributeNames,
                            square=True,
                            cbar_kws={'label': 'Correlation'},
                            ax=ax[i, j],
                            vmin=0,
                            vmax=1,
                            cmap="coolwarm",
                            annot=True)
            ax[i, j].set_title(key, size=12)
            k += 1
    fig.show()


def data_analysis(dataset):
    global attributeNames
    attributeNames = np.asarray(dataset.columns[1:])
    X = dataset.iloc[:, 1:]
    classLabels = dataset.iloc[:, 0]
    # unique class labels
    classNames = np.unique(classLabels)
    # class dictionary
    classDict = dict(zip(classNames, range(len(classNames))))
    # This is the class index vector y:
    y = np.array([classDict[cl] for cl in classLabels])
    # number of data objects and number of attributes
    N, M = X.shape
    # The number of classes, C:
    C = len(classNames)

    corr_all = similarity_analysis(X, M)
    d = {'All': corr_all}
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y == c
        d[classNames[c]] = similarity_analysis(X.iloc[class_mask], M)

    correlation_plot(d)


def outlier(df):
    df.drop(df[df['Height'] > 0.4].index, inplace=True)
    return df


def column_transformer(parameter, x):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), parameter)], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    return x
    
    
"""
if __name__ == "__main__":
    # Import dataset
    dataset = pd.read_csv('abalone.csv')


def feature_scale(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test

    # PCA
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Convert data to float
    Y = np.zeros((len(X), len(X[1]) - 1), float)
    for i in range(len(Y)):
        for f in range(1, len(Y[i]) + 1):
            Y[i][f - 1] = float(X[i][f])

    # Convert age to float
    age = np.zeros(len(y), float)
    for i in range(len(y)):
        age[i] = float(y[i])

    # One of K Encoding
    MFIstr = X[:, 0]
    MFI, b = c2n.categoric2numeric(X[:, 0])
    X = np.hstack((MFI, Y))
    temp = np.vstack((X.T, age))
    X = temp.T

    # Calling the PCA
    pca(X, age, MFIstr)
