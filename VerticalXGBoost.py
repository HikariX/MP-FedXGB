import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
from SSCalculation import *
from Tree import *
import math
import time
np.random.seed(10)
clientNum = 4
class LeastSquareLoss:
    def gradient(self, actual, predicted):
        return -(actual - predicted)

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogLoss():
    def gradient(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob - actual

    def hess(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob * (1.0 - prob) # Mind the dimension

class VerticalXGBoostClassifier:

    def __init__(self, rank, lossfunc, splitclass, _lambda=1, _gamma=0.5, _epsilon=0.1, n_estimators=3, max_depth=3):
        if lossfunc == 'LogLoss':
            self.loss = LogLoss()
        else:
            self.loss = LeastSquareLoss()
        self._lambda = _lambda
        self._gamma = _gamma
        self._epsilon = _epsilon
        self.n_estimators = n_estimators  # Number of trees
        self.max_depth = max_depth  # Maximum depth for tree
        self.rank = rank
        self.trees = []
        self.splitclass = splitclass
        for _ in range(n_estimators):
            tree = VerticalXGBoostTree(rank=self.rank,
                                       lossfunc=self.loss,
                                       splitclass=self.splitclass,
                                       _lambda=self._lambda,
                                        _gamma=self._gamma,
                                       _epsilon=self._epsilon,
                                       _maxdepth=self.max_depth,
                                       clientNum=clientNum)
            self.trees.append(tree)

    def getQuantile(self, colidx):
        split_list = []
        if self.rank != 0: # For client nodes
            data = self.data.copy()
            idx = np.argsort(data[:, colidx], axis=0)
            data = data[idx]
            value_list = sorted(list(set(list(data[:, colidx]))))  # Record all the different value
            hess = np.ones_like(data[:, colidx])
            data = np.concatenate((data, hess.reshape(-1, 1)), axis=1)
            sum_hess = np.sum(hess)
            last = value_list[0]
            i = 1
            if len(value_list) == 1: # For those who has only one value, do such process.
                last_cursor = last
            else:
                last_cursor = value_list[1]
            split_list.append((-np.inf, value_list[0]))
            # if len(value_list) == 15000:
            #     print(self.rank, colidx)
            #     print(value_list)
            while i < len(value_list):
                cursor = value_list[i]
                small_hess = np.sum(data[:, -1][data[:, colidx] <= last]) / sum_hess
                big_hess = np.sum(data[:, -1][data[:, colidx] <= cursor]) / sum_hess
                # print(colidx, self.rank, np.abs(big_hess - small_hess), last, cursor)
                if np.abs(big_hess - small_hess) < self._epsilon:
                    last_cursor = cursor
                else:
                    judge = value_list.index(cursor) - value_list.index(last)
                    if judge == 1: # Although it didn't satisfy the criterion, it has no more split, so we must add it.
                        split_list.append((last, cursor))
                        last = cursor
                    else: # Move forward and record the last.
                        split_list.append((last, last_cursor))
                        last = last_cursor
                        last_cursor = cursor
                i += 1
            if split_list[-1][1] != value_list[-1]:
                split_list.append((split_list[-1][1], value_list[-1]))  # Add the top value into split_list.
            split_list = np.array(split_list)
        return split_list

    def getAllQuantile(self): # Global quantile, must be calculated before tree building, avoiding recursion.
        self_maxlen = 0
        if self.rank != 0:
            dict = {i:self.getQuantile(i) for i in range(self.data.shape[1])} # record all the split
            self_maxlen = max([len(dict[i]) for i in dict.keys()])
        else:
            dict = {}

        recv_maxlen = comm.gather(self_maxlen, root=1)
        maxlen = None
        if self.rank == 1:
            maxlen = max(recv_maxlen)

        self.maxSplitNum = comm.bcast(maxlen, root=1)
        print('MaxSplitNum: ', self.maxSplitNum)
        self.quantile = dict

    def fit(self, X, y):
        data_num = X.shape[0]
        y = np.reshape(y, (data_num, 1))
        y_pred = np.zeros(np.shape(y))
        self.data = X.copy()
        self.getAllQuantile()
        for i in range(self.n_estimators):
            print('In classifier fit, rank: ', self.rank)
            tree = self.trees[i]
            tree.data, tree.maxSplitNum, tree.quantile = self.data, self.maxSplitNum, self.quantile
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(y_and_pred, i)
            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                pass
            else:
                update_pred = tree.predict(X)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred

    def predict(self, X):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main1():
    data = pd.read_csv('./iris.csv').values

    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    train_size_zero = int(zero_data.shape[0] * 0.8)
    train_size_one = int(one_data.shape[0] * 0.8)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0), \
                      np.concatenate((zero_data[train_size_zero:, :-1], one_data[train_size_one:, :-1]), 0)
    y_train, y_test = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1,1), one_data[:train_size_one, -1].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, -1].reshape(-1, 1), one_data[train_size_one:, -1].reshape(-1, 1)), 0)

    X_train_A = X_train[:, 0].reshape(-1, 1)
    X_train_B = X_train[:, 2].reshape(-1, 1)
    X_train_C = X_train[:, 1].reshape(-1, 1)
    X_train_D = X_train[:, 3].reshape(-1, 1)
    X_test_A = X_test[:, 0].reshape(-1, 1)
    X_test_B = X_test[:, 2].reshape(-1, 1)
    X_test_C = X_test[:, 1].reshape(-1, 1)
    X_test_D = X_test[:, 3].reshape(-1, 1)
    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass)

    if rank == 1:
        model.fit(X_train_A, y_train)
        print('end 1')
    elif rank == 2:
        model.fit(X_train_B, np.zeros_like(y_train))
        print('end 2')
    elif rank == 3:
        model.fit(X_train_C, np.zeros_like(y_train))
        print('end 3')
    elif rank == 4:
        model.fit(X_train_D, np.zeros_like(y_train))
        print('end 4')
    else:
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        print('end 0')

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_ori = y_pred.copy()
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        for i in range(y_test.shape[0]):
            print(y_test[i], y_pred[i], y_ori[i])

def main2():
    data = pd.read_csv('./GiveMeSomeCredit/cs-training.csv')
    data.dropna(inplace=True)
    data = data[['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']].values
    # Add features
    # for i in range(99):
    #     data = np.concatenate((data, ori_data[:, 1:]), axis=1)
    data = data / data.max(axis=0)

    ratio = 30000 / data.shape[0]


    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    num = 7500
    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 1:], one_data[train_size_one:train_size_one+int(num * one_ratio), 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 0].reshape(-1, 1),
                                      one_data[train_size_one:train_size_one+int(num * one_ratio), 0].reshape(-1, 1)), 0)

    segment_A = int(0.1*(data.shape[1] - 1))
    segment_B = segment_A + int(0.2*(data.shape[1] - 1))
    segment_C = segment_B + int(0.3*(data.shape[1] - 1))
    X_train_A = X_train[:, 0:segment_A]
    X_train_B = X_train[:, segment_A:segment_B]
    X_train_C = X_train[:, segment_B:segment_C]
    X_train_D = X_train[:, segment_C:]
    X_test_A = X_test[:, :segment_A]
    X_test_B = X_test[:, segment_A:segment_B]
    X_test_C = X_test[:, segment_B:segment_C]
    X_test_D = X_test[:, segment_C:]
    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass)

    start = datetime.now()
    if rank == 1:
        model.fit(X_train_A, y_train)
        end = datetime.now()
        print('In fitting 1: ', end - start)
        time = end - start
        for i in range(5):
            if i == 1:
                pass
            else:
                time += comm.recv(source=i)
        print(time / 5)
        print('end 1')
    elif rank == 2:
        model.fit(X_train_B, np.zeros_like(y_train))
        end = datetime.now()
        comm.send(end - start, dest=1)
        print('In fitting 2: ', end - start)
        print('end 2')
    elif rank == 3:
        model.fit(X_train_C, np.zeros_like(y_train))
        end = datetime.now()
        print('In fitting 3: ', end - start)
        comm.send(end - start, dest=1)
        print('end 3')
    elif rank == 4:
        model.fit(X_train_D, np.zeros_like(y_train))
        end = datetime.now()
        print('In fitting 4: ', end - start)
        comm.send(end - start, dest=1)
        print('end 4')
    else:
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        end = datetime.now()
        print('In fitting 0: ', end - start)
        comm.send(end - start, dest=1)
        print('end 0')

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred2 = y_pred.copy()
        y_pred2[y_pred2 > 0.5] = 1
        y_pred2[y_pred2 <= 0.5] = 0
        y_pred2 = y_pred2.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        result = y_pred2 - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i])

def main3():
    data = np.load('./adult.npy')
    data = data / data.max(axis=0)

    ratio = 0.8

    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]

    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)
    print(train_size_one)
    print(train_size_zero)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:, 1:], one_data[train_size_one:, 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, 0].reshape(-1, 1),
                                      one_data[train_size_one:, 0].reshape(-1, 1)), 0)

    print(y_test.shape)
    print(X_train.shape)
    segment_A = int(0.1 * (data.shape[1] - 1))
    segment_B = segment_A + int(0.2 * (data.shape[1] - 1))
    segment_C = segment_B + int(0.3 * (data.shape[1] - 1))
    print(segment_A, segment_B, segment_C)
    X_train_A = X_train[:, 0:segment_A]
    X_train_B = X_train[:, segment_A:segment_B]
    X_train_C = X_train[:, segment_B:segment_C]
    X_train_D = X_train[:, segment_C:]
    X_test_A = X_test[:, :segment_A]
    X_test_B = X_test[:, segment_A:segment_B]
    X_test_C = X_test[:, segment_B:segment_C]
    X_test_D = X_test[:, segment_C:]
    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass)

    if rank == 1:
        start = datetime.now()
        model.fit(X_train_A, y_train)
        end = datetime.now()
        print('In fitting: ', end - start)
        print('end 1')
    elif rank == 2:
        model.fit(X_train_B, np.zeros_like(y_train))
        print('end 2')
    elif rank == 3:
        model.fit(X_train_C, np.zeros_like(y_train))
        print('end 3')
    elif rank == 4:
        model.fit(X_train_D, np.zeros_like(y_train))
    else:
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        print('end 0')

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_ori = y_pred.copy()
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred2 = y_pred.copy()
        y_pred2[y_pred2 > 0.5] = 1
        y_pred2[y_pred2 <= 0.5] = 0
        y_pred2 = y_pred2.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        result = y_pred2 - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i])

if __name__ == '__main__':
    main1()


