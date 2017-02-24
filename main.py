import pandas as pd
import numpy as np
import math
import random
from sklearn import preprocessing


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# read csv
df = pd.read_csv('C:\\Users\\root\\Downloads\\email_spam.csv')


# convert strings to numerical values
keywords = ['yes', 'no', 'HTML', 'Plain', 'none',
           'big', 'small']

mapping = [1,0,0,1,0,1,2]

df = df.replace(keywords,mapping)

# normalize df
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(np_scaled)


# split to train and test sets for holdout crossvalidation

train_ratio = 0.8
df = df.drop(df.columns[[0]], axis=1)

df.insert(df.shape[1], 'bias', 1)


# shuffle data array
dataset = df.values
random.shuffle(dataset)


response_col = 0

rows = df.shape[0]
trainrows = int(train_ratio*rows)


trainX = dataset[1:trainrows, response_col+1 : dataset.shape[1]]
trainY = dataset[1:trainrows, response_col]

testX = dataset[trainrows:dataset.shape[0], response_col+1: dataset.shape[1]]
testY = dataset[trainrows:dataset.shape[0], response_col]

# here's where the magic happens

epochs = 1000
step_size = 0.01
params_num = testX.shape[1]
params = np.zeros((params_num, 1))
params[:,0] = np.random.uniform(low=-0.5, high=0.5, size=(params_num,))


for i in range(epochs):

    random.shuffle(testX)

    sig_out = [0] * trainX.shape[0]
    diff = [0] * trainX.shape[0]
    gradient = np.zeros((trainX.shape[1], 1))
    data = np.zeros((trainX.shape[1], trainX.shape[0]))
    sig_der = np.zeros((trainX.shape[0], trainX.shape[0]))

    # row iterator loop
    for j in range(trainX.shape[0]):
        # compute gradient vector of negative log likelihood

        # compute sigmoid outputs
        sig_out[j] = sigmoid(np.dot(trainX[j], params[:,]))
        diff[j] = sig_out[j] - trainY[j]


        data[:,j] = trainX[j].transpose()
        gradient[:, 0] = gradient[:,0] + np.multiply(trainX[j].transpose(), diff[j])

    print("Epoch %d" % i)
    print("Train RMSE %0.4f" % np.sqrt(np.dot(diff[j], diff[j]) / len(diff)))
    # compute Hessian
    sig_der = np.diag(np.multiply(sig_out, np.subtract(1, sig_out)))
    hess = np.matmul(np.matmul(data,sig_der), np.transpose(data))

    # invert Hessian
    hess = np.linalg.inv(hess)

    # do the weight update
    params[:,] = params[:,]  - step_size* np.matmul(hess, gradient)

    # do testing
    sig_out_test = [0] * testX.shape[0]
    diff_test = [0] * testX.shape[0]
    for k in range(testX.shape[0]):
        # compute sigmoid outputs
        sig_out_test[k] = sigmoid(np.dot(testX[k], params[:, ]))
        diff_test[k] = sig_out[k] - testY[k]

    print("Test RMSE %0.4f" % np.sqrt(np.dot(diff_test, diff_test) / len(diff_test)))



