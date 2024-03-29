import pickle as pi
import random as r
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import chi2

from toolbox_02450 import train_neural_net

from scipy.stats import t
from scipy.stats import beta as b
from scipy.stats import binom

global data
data = []


def acc_score(y_pred, y_test):
    len_test = len(y_test)
    mix = y_pred + y_test
    mix = list(mix)
    tn = mix.count(0)
    tp = mix.count(2)
    acc = (tn + tp) / len_test
    return 1 - acc


def reg(lambdas, X, Y):
    N, M = X.shape
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    M = M + 1
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X,
                                                                                                      Y,
                                                                                                      lambdas,
                                                                                                      10)
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    Xty = X.T @ Y
    XtX = X.T @ X
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    print("The weights corresponding to " + str(opt_lambda) + " are: ", w_rlr)

    plt.figure(figsize=[12, 6])
    plt.grid()
    plt.plot(lambdas, test_err_vs_lambda)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$", size=16)
    plt.ylabel(r"$E_{gen}$", size=16)
    plt.show()


def lin_reg(x_train, y_train, x_test, y_test):
    N, M = x_train.shape
    x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), 1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), 1)
    M = M + 1
    param = np.power(10., range(-5, 5))
    opt_val_err, p, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(x_train,
                                                                                             y_train,
                                                                                             param,
                                                                                             10)
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = p * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    Xty = x_train.T @ y_train
    XtX = x_train.T @ x_train
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    # error_train_rlr = np.square(y_train - x_train @ w_rlr).sum(axis=0) / y_train.shape[0]
    store_error("lin", test_err_vs_lambda)
    err = np.square(y_test - x_test @ w_rlr).sum(axis=0) / y_test.shape[0]
    return err, p


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


def knn(x_train, y_train, x_test, y_test, param):
    classifier = KNeighborsClassifier(n_neighbors=param)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    global cB
    if a == 1:
        cB.append(y_pred == y_test)
        print("Knn")
    return acc_score(y_test, y_pred)


def log_reg(x_train, y_train, x_test, y_test, param):
    classifier = LogisticRegression(random_state=0, C=1 / param)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test).T
    global cA
    if a == 1:
        cA.append(y_pred == y_test)
    return acc_score(y_pred, y_test)


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
    # plt.figure()
    # plt.plot(y_test, 'ok')
    # plt.plot(y_test_est_n, 'or')
    # plt.show()

    return loss


def cross_validation(X, Y, model, param, K):
    CV = model_selection.KFold(K, shuffle=True)
    err = np.zeros([K, len(param)])
    par = np.zeros([K, len(param)])
    for i, (train_index, test_index) in enumerate(CV.split(X, X)):
        for k in range(len(param)):
            x_train = X[train_index, :]
            y_train = Y[train_index]
            x_test = X[test_index, :]
            y_test = Y[test_index]

            # Train the network
            # error_rate = eval(model1)(X_train, Y_train,X_test, Y_test,param1)
            temp = model(x_train, y_train, x_test, y_test, param[k])
            try:
                err[i, k] = temp[0]
                par[i, k] = temp[1]
            except:
                err[i, k] = temp
        # weights = [net[i].weight.data.numpy().T for i in [0, 2]]
        # biases = [net[i].bias.data.numpy() for i in [0, 2]]
        # tf = [str(net[i]) for i in [1, 2]]
        # draw_neural_net(weights, biases, tf)
    er_gen = np.mean(err, 0)
    print(err)
    print(param[np.argmin(er_gen)])
    if isinstance(param[0], int) and model == neural_network_train:
        plt.figure(1)
        plt.plot(param, er_gen, 'k')
        plt.title("Loss of Neural Network")
        plt.xlabel("Number of nodes in the hidden layer")
        plt.ylabel("Loss")
        store_error("ann", er_gen)
    elif model == knn:
        store_error("knn", er_gen)
    elif model == log_reg:
        store_error("log", er_gen)
    else:
        plt.figure(2)
        plt.plot(er_gen, 'k')
    plt.show()
    return param[np.argmin(er_gen)], err, par


def baseline_reg(y_train, y_test):
    return np.square(y_test - np.ones(len(y_test)) * np.mean(y_train)).sum(axis=0) / y_test.shape[0]


def baseline_class(y_train, y_test):
    if sum(y_train) > len(y_train):
        y_pred = np.ones(len(y_test))
    elif sum(y_train) == len(y_train):
        y_pred = r.randint(0, 1) * np.ones(len(y_test))
    else:
        y_pred = np.zeros(len(y_test))
    print("Baseline out")
    global cC
    if a == 1:
        cC.append(y_pred == y_test)
        print("Baseline")
    return acc_score(y_test, y_pred)


def models(x_train, y_train, x_test, y_test, model_indices):
    global a
    K = 10
    if model_indices == "ann":
        model = neural_network_train
        K = 5
        param = (1, 2, 3, 4, 5, 7, 10)
    elif model_indices == "knn":
        param = range(1, 100)
        model = knn
    elif model_indices == "knn_loo":
        param = (1, 10, 50, 100)
        model = knn
        K = len(x_train) - 1
    elif model_indices == "lin":
        return lin_reg(x_train, y_train, x_test, y_test)
    elif model_indices == "log":
        model = log_reg
        param = np.power(10., range(-5, 5))
    elif model_indices == "reg_baseline":
        return baseline_reg(y_train, y_test)
    elif model_indices == "class_baseline":
        a = 1
        return baseline_class(y_train, y_test)
    else:
        print("Model name does not exist!")
        return None

    a = 0
    p, _, _ = cross_validation(x_train, y_train, model, param, K)
    a = 1
    err = model(x_train, y_train, x_test, y_test, p)
    a = 0
    # store_error("Outer",p)
    # Plotting

    return err, p


def column_transformer(param, X):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), param)], remainder='passthrough')
    return np.array(ct.fit_transform(X))


def feature_scale(x):
    sc = StandardScaler()
    x_trans = sc.fit_transform(x)
    return x_trans


def significant(z, alpha, method):
    J = len(z)
    # K=J
    zhat = np.mean(z)
    s2 = 1 / (J - 1) * sum((z - zhat) ** 2)
    sigma = np.sqrt(s2 * (1 / J + 1 / (J - 1)))

    if method == "1sided":
        # zstar=t.interval(1-alpha, len(z)-1, loc=0, scale=sigma)[1]
        p = 1 - t.cdf(abs(zhat), len(z) - 1, loc=0, scale=sigma)
        print("The p-value is ", p)
        if p > alpha:
            print("H0 cannot be rejected")
        if p < alpha:
            if zhat > 0:
                print("H0 rejected and H1 (mean(z)>0) accepted with " + str(1 - alpha) + " confidence level")
            else:
                print("H0 rejected and H1 (mean(z)<0)accepted with " + str(1 - alpha) + " confidence level")
    if method == "2sided":
        # tstar=t.interval(1-alpha, len(z)-1, loc=0, scale=sigma)[1]
        p = 2 * t.cdf(-abs(zhat), len(z) - 1, loc=0, scale=sigma)
        bounds = sigma * t.ppf(alpha / 2, J - 1)
        CI = [zhat + bounds, zhat - bounds]
        print("The " + str((1 - alpha) * 100) + "% CI is: ", CI)
        print(zhat + bounds)
        print("The p-value is ", p)
        if abs(p) > alpha / 2:
            print("H0 cannot be rejected")
        if p < 0.05:
            print(r"H0 rejected and H1 (mean(z) not 0) accepted with {} confidence level".format(str(1 - alpha)))


def store_error(method, err):
    d = {"method": method, "err": err}
    data.append(d)


def get_error(method):
    d = []
    for i in data:
        if i["method"] == method:
            d.append(i["err"])
    return np.stack(d, axis=0)


def mcnemars(c1, c2, alpha):
    d1 = 0
    d2 = 0
    n = len(np.hstack(c1))
    for i in range(len(c1)):
        for f in range(len(c1[i])):
            if c1[i][f] and not c2[i][f]:
                d1 += 1
            if not c1[i][f] and c2[i][f]:
                d2 += 1
    print("b = " + str(d1) + " c = " + str(d2))
    N = d1 + d2
    if N < 5:
        print("warning, n12+n21<5 ")
    E = (d1 - d2) / n  # expected value
    Q = (n ** 2 * (n + 1) * (E + 1) * (E - 1)) / (n * (d1 + d2) - (d1 - d2) ** 2)
    f = (E + 1) * (Q - 1) / 2
    g = (1 - E) * (Q - 1) / 2
    CI1 = 2 * b.ppf(alpha / 2, f, g) - 1
    CI2 = 2 * b.ppf(1 - alpha / 2, f, g) - 1
    CI = [CI1, CI2]
    p = 2 * binom.cdf(min([d1, d2]), n=N, p=0.5)
    print("The " + str((1 - alpha) * 100) + "% CI is: ", CI)
    print("The p-value is ", p)


def prepare_table(error, param, algo):
    if len(algo) == 5:
        table = pd.DataFrame()
        table[algo[0]] = error[:, 0]
        table[algo[1]] = param[:, 1]
        table[algo[2]] = error[:, 1]
        table[algo[3]] = param[:, 2]
        table[algo[4]] = error[:, 2]
        return table


def plotting(method):
    temp = get_error(method)
    if method == "ann":
        param = (1, 2, 3, 4, 5, 7, 10)
    elif method == "knn":
        param = range(1, 100)
    elif method == "lin":
        param = np.power(10., range(-5, 5))
        temp = np.mean(temp, 0)
    elif method == "log":
        param = np.power(10., range(-5, 5))
    else:
        return None

    plt.figure(figsize=[12, 6])
    for i in temp:
        plt.plot(param, i, 'or')
    plt.plot(param, np.mean(temp, 0), 'k')
    if method == "log" or method == "lin":
        plt.xscale("log")
    plt.xlabel(r"Model complexity controlling parameter", size=16)
    plt.ylabel(r"$E_{test}$", size=16)
    plt.grid()
    plt.show()
    # plt.title("Loss of Neural Network")
    # plt.xlabel("Number of nodes in the hidden layer")
    # plt.ylabel("Loss")


def read_pickle(filename):
    pickle_file = open(filename, "rb")
    load_file = pi.load(pickle_file)
    pickle_file.close()
    return load_file


if __name__ == "__main__":
    # Importing the dataset
    plt.close(fig='all')
    cA = []
    cB = []
    cC = []
    dataset = pd.read_csv('abalone.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    Y_class = Y.copy()
    Y_class[Y_class <= Y_class.mean()] = 0
    Y_class[Y_class > 0] = 1

    # Create a age column from the Rings column + 1.5
    dataset['Age'] = dataset['Rings'] + 1.5
    dataset.drop('Rings', axis=1, inplace=True)

    # Column transform Sex column
    X = column_transformer([0], X)

    # Feature scale
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

    # reg(np.power(10., range(-10, 9)), X_float, Y_float)
    # print(cross_validation(X, age, neural_network_train, [5, 6, 7], 5))

    choice = 3
    if choice == 1:
        # mb1, err1, par1 = cross_validation(X_float, Y_float, models, ["reg_baseline", "lin", "ann"], 5)
        mb2, err2, par2 = cross_validation(X_float, Y_class, models, ["class_baseline", "knn", "log"], 10)
        # table_reg = prepare_table(err1, par1, ["baseline_error", "lin_param", "lin_error", "ann_param", "ann_error"])
        # table_class = prepare_table(err2, par2, ["baseline_error", "knn_param", "knn_error", "log_param", "log_error"])
        abalone2 = {"data": data, "mb2": mb2, "err2": err2, "par2": par2, "ca": cA, "cb": cB, "cc": cC}
        pickle_out = open("abalone2.pickle", "wb")
        pi.dump(abalone2, pickle_out)
        pickle_out.close()
    elif choice == 2:
        file = read_pickle("abalone.pickle")
        table_reg = prepare_table(file["err1"], file["par1"],
                                  ["baseline_error", "lin_param", "lin_error", "ann_param", "ann_error"])
        data = file["data"]
        plotting("ann")
        plotting("lin")
    elif choice == 3:
        file = read_pickle("abalone2.pickle")
        table_class = prepare_table(file["err2"], file["par2"],
                                    ["baseline_error", "knn_param", "knn_error", "log_param", "log_error"])
        data = file["data"]
        cA = file["ca"]
        cB = file["cb"]
        cC = file["cc"]
        plotting("knn")
        plotting("log")
