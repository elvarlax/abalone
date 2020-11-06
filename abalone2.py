import random as r
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from toolbox_02450 import train_neural_net

def reg(lambdas,X,Y):
    N, M = X.shape
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    M = M+1
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, Y, lambdas, 10)
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    Xty = X.T @ Y
    XtX = X.T @ X
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    
    print("The weights corresponding to "+str(opt_lambda)+" are: ",w_rlr)
    
    plt.figure(figsize=[12,6])
    plt.grid()
    plt.plot(lambdas,test_err_vs_lambda)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$",size=16)
    plt.ylabel(r"$E_{gen}$",size=16)
    plt.show()
   
    #return w_rlr
    
def lin_reg(x_train, y_train, x_test, y_test):
    N, M = x_train.shape
    x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), 1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), 1)
    M = M + 1
    param = np.power(10., range(-10, 9))
    opt_val_err, p, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(x_train, y_train, param,
                                                                                             10)
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = p * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    Xty = x_train.T @ y_train
    XtX = x_train.T @ x_train
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr = np.square(y_train - x_train @ w_rlr).sum(axis=0) / y_train.shape[0]
    err = np.square(y_test - x_test @ w_rlr).sum(axis=0) / y_test.shape[0]
    return p, err


def rlr_validate(X, y, lambdas, cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        # mu = np.mean(X_train[:, 1:], 0)
        # sigma = np.std(X_train[:, 1:], 0)

        # X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        # X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(axis=0)
            test_error[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda


# (X,y,param)
def knn(x, y, param):
    # Training the K-NN model on the Training set
    # Euclidean distance between neighbors of 5
    classifier = KNeighborsClassifier(n_neighbors=param, metric='minkowski', p=2)
    classifier.fit(x, y)
    # Predicting the Test set results
    y_pred = classifier.predict(x)
    return accuracy_score(y, y_pred)


def logistic_regression(x_train, y_train, x_test, y_test, param):
    pass


def neural_network_train(x_train, y_train, x_test, y_test, param):
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    global model, loss_fn

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(9, param),  # M features to H hiden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(param, 1),  # H hidden units to 1 output neuron
        # torch.nn.Sigmoid() # final tranfer function
    )

    loss_fn = torch.nn.MSELoss()

    global net, final_loss, learning_curve

    max_iter = 10000

    y_train = y_train.unsqueeze(1)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=x_train,
                                                       y=y_train,
                                                       n_replicates=1,
                                                       max_iter=max_iter)

    y_test_est = net(x_test)  # activation of final note, i.e. prediction of network
    # y_test_est = (y_sigmoid > .5)#._cast_uint8_t() # threshold output of sigmoidal function
    # Determine errors and error rate
    # e = (y_test_est != Y_test)
    # error_rate = (sum(e).type(torch.float)/len(Y_test)).data.numpy()

    y_test_est_n = y_test_est.detach().numpy()
    y_test = y_test.unsqueeze(1)
    loss = loss_fn(y_test, y_test_est)
    plt.figure()
    plt.plot(y_test, 'ok')
    plt.plot(y_test_est_n, 'or')
    plt.show()

    return loss


def cross_validation(X, Y, model, param, K):
    CV = model_selection.KFold(K, shuffle=True)
    err = np.zeros([K, len(param)])
    for i, (train_index, test_index) in enumerate(CV.split(X, X)):
        for k in range(len(param)):
            x_train = X[train_index, :]
            y_train = Y[train_index]
            x_test = X[test_index, :]
            y_test = Y[test_index]

            # Train the network
            # error_rate = eval(model1)(X_train, Y_train,X_test, Y_test,param1)
            err[i, k] = model(x_train, y_train, x_test, y_test, param[k])

        # weights = [net[i].weight.data.numpy().T for i in [0, 2]]
        # biases = [net[i].bias.data.numpy() for i in [0, 2]]
        # tf = [str(net[i]) for i in [1, 2]]
        # draw_neural_net(weights, biases, tf)
    er_gen = np.mean(err, 0)
    print(err)
    return param[np.argmin(er_gen)]


def baseline_reg(y_train, y_test):
    loss = np.square(y_test - np.ones(len(y_test)) * np.mean(y_train)).sum(axis=0) / y_test.shape[0]
    return loss


def baseline_class(y_train, y_test):
    if sum(y_train) > len(y_train):
        y_pred = np.ones(len(y_test))
    elif sum(y_train) == len(y_train):
        y_pred = r.randint(0, 1) * np.ones(len(y_test))
    else:
        y_pred = np.zeros(len(y_test))
    return accuracy_score(y_test, y_pred)


def models(x_train, y_train, x_test, y_test, model_indices):
    if model_indices == "ann":
        model = neural_network_train
        param = (5, 7, 9)
    elif model_indices == "knn":
        param = (1, 5, 10)
        chosen_k = [knn(x_train, y_train, k) - 1 for k in param]
        err = knn(x_test, y_test, np.argmin(chosen_k))
        return err
    elif model_indices == "lin":
        p, err = lin_reg(x_train, y_train, x_test, y_test)
        return err
    elif model_indices == "log":
        pass
    elif model_indices == "reg_baseline":
        err = baseline_reg(y_train, y_test)
        return err
    elif model_indices == "class_baseline":
        
        err = baseline_class(y_train, y_test)
        return err
    else:
        print("Model name does not exist!")
        return None

    p = cross_validation(x_train, y_train, model, param, 5)
    err = model(x_train, y_train, x_test, y_test, p)

    return err


def column_transformer(param, X):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), param)], remainder='passthrough')
    return np.array(ct.fit_transform(X))


def feature_scale(x):
    sc = StandardScaler()
    x_trans = sc.fit_transform(x)
    # x_test = sc.transform(x_test)
    return x_trans


if __name__ == "__main__":
    # Importing the dataset
    plt.close(fig='all')
    dataset = pd.read_csv('abalone.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    # Create a age column from the Rings column + 1.5
    dataset['Age'] = dataset['Rings'] + 1.5
    dataset.drop('Rings', axis=1, inplace=True)

    # Column transform Sex column
    X = column_transformer([0], X)

    cross_validation(X, Y, models, ["knn"], 10)

    X = feature_scale(X)
    Y = feature_scale(Y.reshape(-1, 1))

    X_float = np.zeros((len(X), len(X[1]) - 1), float)

    for i in range(len(X)):
        for f in range(1, len(X[i])):
            X_float[i][f - 1] = float(X[i][f])

    # Convert age to float
    Y_float = np.zeros(len(Y), float)
    for i in range(len(Y)):
        Y_float[i] = float(Y[i])
    reg(np.power(10.,range(-10,9)),X_float,Y_float)

    cross_validation(X, Y, models, ["class_baseline", "log", "knn"], 10)
    cross_validation(X_float, Y_float, models, ["reg_baseline", "lin", "ann"], 2)
