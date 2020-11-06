import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from toolbox_02450 import train_neural_net


def knn(x_train, y_train, x_test, y_test, param):
    # Training the K-NN model on the Training set
    # Euclidean distance between neighbors of 5
    classifier = KNeighborsClassifier(n_neighbors=param, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)


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


def models(x_train, y_train, x_test, y_test, model_indices):
    if model_indices == "ann":
        model = neural_network_train
        param = (5, 7, 9)
    elif model_indices == "knn":
        model = knn
        param = (1, 5, 10)
    elif model_indices == "lin":
        pass
    elif model_indices == "log":
        pass
    else:
        print("Model name does not exist!")
        return None

    p = cross_validation(x_train, y_train, model, param, 10)

    err = model(x_train, y_train, x_test, y_test, p)

    return err


def column_transformer(param, X):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), param)], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    return X


def feature_scale(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test


if __name__ == "__main__":
    # Importing the dataset
    dataset = pd.read_csv('abalone.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    # Create a age column from the Rings column + 1.5
    dataset['Age'] = dataset['Rings'] + 1.5
    dataset.drop('Rings', axis=1, inplace=True)

    # Column transform Sex column
    X = column_transformer([0], X)

    # Training the K-NN model on the Training set
    cross_validation(X, Y, knn, [1, 5, 10], 5)

    X_ann = dataset.iloc[:, :-1].values
    X_ann = column_transformer([0], X_ann)
    X = np.zeros((len(X_ann), len(X_ann[1]) - 1), float)

    for i in range(len(X)):
        for f in range(1, len(X[i]) + 1):
            X[i][f - 1] = float(X_ann[i][f])

    # Convert age to float
    age = np.zeros(len(Y), float)
    for i in range(len(Y)):
        age[i] = float(Y[i]) / max(Y)

    # print(cross_validation(X, age, neural_network_train, [5, 6, 7], 5))

    cross_validation(X, age, models, [0], 2)
