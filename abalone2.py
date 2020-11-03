# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:19:10 2020

@author: Simon Herlett
"""

import torch
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import categoric2numeric as c2n
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary

def NeuralNetworkTrain(X_train,Y_train,X_test,Y_test,parameter1):
    
    global model, loss_fn
    
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(10, parameter1), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(parameter1, 1), # H hidden units to 1 output neuron
                    #torch.nn.Sigmoid() # final tranfer function
                    )

    loss_fn = torch.nn.MSELoss()
    
    global net, final_loss, learning_curve
    
    max_iter = 10000
    
    if len(Y_train)  != 1:
        Y_train = Y_train.T
        
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=Y_train,
                                                       n_replicates=1,
                                                       max_iter=max_iter)
    
    y_test_est = net(X_test) # activation of final note, i.e. prediction of network
    #y_test_est = (y_sigmoid > .5)#._cast_uint8_t() # threshold output of sigmoidal function
    # Determine errors and error rate
    #e = (y_test_est != Y_test)
    #error_rate = (sum(e).type(torch.float)/len(Y_test)).data.numpy()
    
    y_test_est = y_test_est.detach().numpy()
    plt.figure()
    plt.plot(Y_test, 'ok')
    plt.plot(y_test_est,'or')
    plt.show()
    
    return final_loss
    
    
def CrossValidation(X,Y,model1,param1,K):  
    
    CV = model_selection.KFold(K,shuffle=True)
    for p in param1:
        for  k, (train_index, test_index) in enumerate(CV.split(X,X)):
            
            X_train = torch.Tensor(X[train_index,:] )
            Y_train = torch.Tensor(Y[train_index] )
            X_test = torch.Tensor(X[test_index,:] )
            Y_test = torch.Tensor(Y[test_index] )
        
            # Train the network
            #error_rate = eval(model1)(X_train, Y_train,X_test, Y_test,param1)
            er = model1(X_train, Y_train,X_test, Y_test,p)
            print(er)
        
        #weights = [net[i].weight.data.numpy().T for i in [0,2]]
        #biases = [net[i].bias.data.numpy() for i in [0,2]]
        #tf =  [str(net[i]) for i in [1,2]]
        #draw_neural_net(weights, biases, tf)
        
# Testing
if __name__ == "__main__":
    dataset = pd.read_csv('abalone.csv')
    dataset['Age'] = dataset['Rings'] + 1.5
    dataset.drop('Rings', axis=1, inplace=True)
    
    Xtemp = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    
    X = np.zeros((len(Xtemp), len(Xtemp[1]) - 1), float)
    for i in range(len(X)):
        for f in range(1, len(X[i]) + 1):
            X[i][f - 1] = float(Xtemp[i][f])
                
    MFIstr = Xtemp[:, 0]
    MFI, b = c2n.categoric2numeric(Xtemp[:, 0])
    X = np.hstack((MFI, X))
    
    # Convert age to float
    age = np.zeros(len(Y), float)
    for i in range(len(Y)):
        age[i] = float(Y[i])/max(Y)
        
        
    CrossValidation(X,age,NeuralNetworkTrain,6,3)
        
        
