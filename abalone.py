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
    
    Y = Y - np.ones((len(Y), 1)) * Y.mean(0) # Translate the data to around origin
    Y = Y / (np.ones((len(Y), 1)) * Y.std(0))
    U, S, V = linalg.svd(Y, full_matrices=False) # 

    rho = S / sum(S)
    rhoa = np.zeros((len(S),), float)
    for i in range(len(rho)):
        rhoa[i] = sum(rho[:(i + 1)])

        rhoa[i] = sum(rho[:(i+1)])
    
    K = 4;
    plt.figure()
    plt.plot(rhoa, 'o-')
    plt.plot(rho, 'o-')
    # plt.legend(["","Variation explained by the given principle component"])
    plt.show()

    Shat = np.vstack((np.diag(S[0:K]), np.zeros((L - K, K), float)))
    Vhat = V[0:K, :]
    print(str(Vhat))
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
        plt.plot(Xhat[i, 2], Xhat[i, 3], st, color=(min(age[i] / 15, 1), 0, max(1 - age[i] / 15, 0)))
    plt.xlabel("PCA #3")
    plt.ylabel("PCA #4")
    plt.show()
    
    plt.figure()
    for i in range(len(age)):
        if MFI[i] == 'M':
            st = 'o'
        elif MFI[i] == 'F':
            st = 'x'
        else:
            st = '+'
        plt.plot(Xhat[i, 0], Xhat[i, 3], st, color=(min(age[i] / 15, 1), 0, max(1 - age[i] / 15, 0)))
    plt.xlabel("PCA #1")
    plt.ylabel("PCA #4")
    plt.show()
    
    v = np.vstack((MFI,Xhat[:,:K].T))
    df2 = pd.DataFrame(v.T,
                   columns=['Sex','PCA1', 'PCA2', 'PCA3', 'PCA4'])
    plt.figure()
    sns.pairplot(df2,hue = 'Sex', palette="Set1", markers=["s", "o", "D"])
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

def similarity_analysis(X,M,method='cor'):
    sim_mat= np.zeros((M, M), float)
    for j in range(M):
        for i in range(M):
            sim_mat[i,j]=sim.similarity(X.iloc[:,i],X.iloc[:,j],method)
    #heatmap_plot(sim_mat)
    return sim_mat

def heatmap_plot(corr_data):
    plt.figure()
    #masking unique values
    mask = np.zeros_like(corr_data)
    mask[np.triu_indices_from(mask)] = True    
    sns.heatmap(corr_data,mask=mask,xticklabels=attributeNames, yticklabels=attributeNames,square=True,
                cbar_kws={'label': 'Correlation'})
    plt.xlabel('Attribute name')
    plt.ylabel('Attribute name')
    plt.show()
    

def correlation_plot(d):
  nrows,ncols=int(len(d)/2),int(len(d)/2)
  fig, ax = plt.subplots(nrows, ncols,sharex=True,sharey=True)
  k=0
  for i in range(nrows):
    for j in range(ncols):
      key=list(d.keys())[k]
      corr_data=d[key]
      mask = np.zeros_like(corr_data)
      mask[np.triu_indices_from(mask)] = True
      if k==0 or k==2:
          sns.heatmap(corr_data,mask=mask,xticklabels=attributeNames, yticklabels=attributeNames,square=True,ax=ax[i, j],
                      vmin=0, vmax=1, cmap="coolwarm",annot=True)
      else:
          sns.heatmap(corr_data,mask=mask,xticklabels=attributeNames, yticklabels=attributeNames,square=True,
                cbar_kws={'label': 'Correlation'},ax=ax[i, j],vmin=0, vmax=1, cmap="coolwarm",annot=True)
      #heatmap_plot(d[key])
      #if k==0:
        #ax[i,j].legend(loc='upper right')
        #ax[i,j].set_ylabel('Attribute name')
      #if i==1:
        #ax[i,j].set_xlabel('Attribute name')
      #if j==0:
          #ax[i,j].set_ylabel('Attribute name')
      ax[i,j].set_title(key,size=12)
      k+=1

  #fig.tight_layout()
  fig.show()
    
def data_analysis(dataset):
    global attributeNames
    attributeNames = np.asarray(dataset.columns[1:])
    X = dataset.iloc[:, 1:]
    classLabels = dataset.iloc[:,0]
    #unique class labels 
    classNames = np.unique(classLabels)
    #class dictionary
    classDict = dict(zip(classNames,range(len(classNames))))
    #This is the class index vector y:
    y = np.array([classDict[cl] for cl in classLabels])
    #number of data objects and number of attributes 
    N, M = X.shape
    # The number of classes, C:
    C = len(classNames)
    
    
    corr_all=similarity_analysis(X,M)
    d = {'All':corr_all}
    for c in range(C):
    # select indices belonging to class c:
        class_mask = y==c
        d[classNames[c]]=similarity_analysis(X.iloc[class_mask],M)
        
    
    correlation_plot(d)
        
    #return X, C,y


def outlier(df):
    df.drop(df[df['Height']>0.4].index,inplace=True)
    return df
        
    
#def visual_data(dataset):
    
    #df = dataset[dataset.columns[1:-1]]
    #box_plot(dataset['Sex'], dataset['Age'], df)
    #matrix_plot(df)
    
    
if __name__ == "__main__":
    # Import dataset
    dataset = pd.read_csv('abalone.csv') 
    
    # Create a age column from the Rings column + 1.5
    dataset['Age'] = dataset['Rings'] + 1.5
    dataset.drop('Rings', axis=1, inplace=True)
    
    #matrix_plot(dataset)
    
    #remove outliers
    dataset=outlier(dataset)
    #dataset.iloc[:,0:-1].boxplot()
    
############# put this in a pca function that manages the pca analysis??????????????????  
    X = dataset.iloc[:, :-1].values

    y = dataset.iloc[:, -1].values

    # Move this to a function
    Y = np.zeros((len(X), len(X[1]) - 1), float)
    for i in range(len(Y)):
        for f in range(1, len(Y[i])+1):
            Y[i][f-1] = float(X[i][f])
 
    age = np.zeros(len(y), float)
    for i in range(len(y)):
        age[i] = float(y[i])

    MFIstr = X[:, 0]
    MFI, b = c2n.categoric2numeric(X[:, 0])
    X = np.hstack((MFI, Y))
    
    ###############################function calling

    # pca(Y, y, MFI)
        
    #pca(X,age, MFIstr)
    
    #visual_data(dataset)

    #data_analysis(dataset)



    
    # matrix_plot(dataset)
    
    #temp = np.vstack((X.T,age))
   # X = temp.T
    #pca(X, age, MFIstr)
