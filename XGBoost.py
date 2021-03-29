import numpy as np
import progressbar
import pandas as pd
from datetime import *
np.random.seed(10)
class LeastSquareLoss():
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

class Tree:
    def __init__(self, value=None, leftBranch=None, rightBranch=None, col=-1, result=None):
        self.value = value
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.col = col
        self.result = result

class XGBoostTree:
    def __init__(self, lossfunc, _lambda, _gamma, _max_depth, _epsilon=0.05):
        self.loss = lossfunc
        self._lambda = _lambda
        self._gamma = _gamma
        self._epsilon = _epsilon
        self._max_depth = _max_depth

    def _split(self, y):
        y, y_pred = y[:, 0].reshape(-1, 1), y[:, 1].reshape(-1, 1)
        return y, y_pred

    def _leaf_gain(self, g, h):
        nominator = np.power(g, 2)
        denominator = h + self._lambda
        return nominator / denominator

    def _split_criterion(self, left_g, left_h, right_g, right_h):
        gain = (self._leaf_gain(left_g, left_h) + self._leaf_gain(right_g, right_h) -
                self._leaf_gain(left_g + right_g, left_h + right_h)) * 0.5 - self._gamma
        # print('In split criterion')
        # print(self._leaf_gain(left_g, left_h))
        # print(self._leaf_gain(right_g, right_h))
        # print(self._leaf_gain(left_g + right_g, left_h + right_h))
        # print('Out split criterion')
        return gain

    def getQuantile(self, colidx, X, y, y_pred):
        split_list = []
        data = np.concatenate((X, y, y_pred), axis=1)
        data = data.copy()
        idx = np.argsort(data[:, colidx], axis=0)
        data = data[idx]
        value_list = sorted(list(set(list(data[:, colidx]))))  # Record all the different value
        hess = np.ones_like(data[:, colidx])
        data = np.concatenate((data, hess.reshape(-1, 1)), axis=1)
        sum_hess = np.sum(hess)
        last = value_list[0]
        i = 1
        if len(value_list) == 1:  # For those who has only one value, do such process.
            last_cursor = last
        else:
            last_cursor = value_list[1]
        split_list.append((-np.inf, value_list[0]))
        while i < len(value_list):
            cursor = value_list[i]
            small_hess = np.sum(data[:, -1][data[:, colidx] <= last]) / sum_hess
            big_hess = np.sum(data[:, -1][data[:, colidx] <= cursor]) / sum_hess
            # print(colidx, self.rank, np.abs(big_hess - small_hess), last, cursor)
            if np.abs(big_hess - small_hess) < self._epsilon:
                last_cursor = cursor
            else:
                judge = value_list.index(cursor) - value_list.index(last)
                if judge == 1:  # Although it didn't satisfy the criterion, it has no more split, so we must add it.
                    split_list.append((last, cursor))
                    last = cursor
                else:  # Move forward and record the last.
                    split_list.append((last, last_cursor))
                    last = last_cursor
                    last_cursor = cursor
            i += 1
        if split_list[-1][1] != value_list[-1]:
            split_list.append((split_list[-1][1], value_list[-1]))  # Add the top value into split_list.
        split_list = np.array(split_list)
        return split_list

    def getAllQuantile(self, X, y): # Global quantile, must be calculated before tree building, avoiding recursion.
        y, y_pred = self._split(y)
        column_length = X.shape[1]
        dict = {i: self.getQuantile(i, X, y, y_pred) for i in range(column_length)}  # record all the split
        self.quantile = dict

    def buildTree(self, X, y, depth=1):
        data = np.concatenate((X, y), axis=1)
        y, y_pred = self._split(y)
        column_length = X.shape[1]
        gradient = self.loss.gradient(y, y_pred) # Calculate the loss at begining, avoiding later calculation.
        hessian = self.loss.hess(y, y_pred)
        # print('*' * 10, 'Gradient')
        # print(np.concatenate([gradient, hessian], axis=1)[:20])
        G = np.sum(gradient)
        H = np.sum(hessian)
        if depth > self._max_depth:
            return Tree(result=- G / (H + self._lambda))

        bestGain = 0
        bestSplit = 0
        bestSet = ()

        for col in range(column_length):
            splitList = self.quantile[col]
            GL = 0
            HL = 0
            for k in range(splitList.shape[0]):
                left = splitList[k][0]
                right = splitList[k][1]
                idx = ((data[:, col] <= right) & (data[:, col] > left))
                GL += np.sum(gradient[idx])
                HL += np.sum(hessian[idx])
                GR = G - GL
                HR = H - HL
                gain = self._split_criterion(GL, HL, GR, HR)
                if gain > bestGain:
                    bestGain = gain
                    bestSplit = (col, right)
                    bestSet = (data[data[:, col] <= right], data[data[:, col] > right])

        if bestGain > 0:
            # print('Split value: ', bestSplit[1])
            # print('Into left')
            leftBranch = self.buildTree(bestSet[0][:, :-2], bestSet[0][:, -2:], depth + 1)
            # print('Out left')
            # print('Into right')
            rightBranch = self.buildTree(bestSet[1][:, :-2], bestSet[1][:, -2:], depth + 1)
            # print('Out right')
            return Tree(value=bestSplit[1], leftBranch=leftBranch, rightBranch=rightBranch, col=bestSplit[0])
        else:
            # print(-G/(H + self._lambda))
            return Tree(result=- G / (H + self._lambda))

    def fit(self, X, y):
        self.getAllQuantile(X, y)
        self.Tree = self.buildTree(X, y)

    def classify(self, tree, data):
        if tree.result != None:
            return tree.result
        else:
            branch = None
            v = data[tree.col]
            if isinstance(v, int) or isinstance(v, float):
                if v > tree.value:
                    branch = tree.rightBranch
                else:
                    branch = tree.leftBranch
            return self.classify(branch, data)

    def predict(self, data):
        data_num = data.shape[0]
        result = []
        for i in range(data_num):
            result.append(self.classify(self.Tree, data[i]))
        result = np.array(result).reshape((-1, 1))
        return result


class XGBoostClassifier:
    def __init__(self, lossfunc, _lambda=1, _gamma=0.5, _epsilon=0.1, n_estimators=3, learning_rate=1, min_samples_split=2, max_depth=3):
        if lossfunc == 'LogLoss':
            self.loss = LogLoss()
        else:
            self.loss = LeastSquareLoss()
        self._lambda = _lambda
        self._gamma = _gamma
        self._epsilon = _epsilon
        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.max_depth = max_depth  # Maximum depth for tree
        self.bar = progressbar.ProgressBar()
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostTree(
                lossfunc=self.loss,
                _lambda=self._lambda,
                _gamma=self._gamma,
                _max_depth=self.max_depth,
                _epsilon=self._epsilon,)
            self.trees.append(tree)

    def fit(self, X, y):
        data_num = X.shape[0]
        y = np.reshape(y, (data_num, 1))
        y_pred = np.zeros(np.shape(y))
        # y_pred = np.random.rand(y.shape[0]).reshape(-1, 1)
        for i in range(self.n_estimators):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            # print(y_and_pred)
            tree.fit(X, y_and_pred)
            print('-' * 100)
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (data_num, 1))
            y_pred += update_pred

    def predict(self, X):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (data_num, 1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            y_pred += update_pred
        return y_pred

def main():
    data = pd.read_csv('./filtered_data_median.csv').values
    # train_size = int(data.shape[0] * 0.9)
    # X_train, X_test = data[:train_size, :-2], data[train_size:, :-2]
    # y_train, y_test = data[:train_size, -2].reshape(-1, 1), data[train_size:, -2].reshape(-1, 1)

    zero_index = data[:, -2] == 0
    one_index = data[:, -2] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    train_size_zero = int(zero_data.shape[0] * 0.8)
    train_size_one = int(one_data.shape[0] * 0.8)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, :-2], one_data[:train_size_one, :-2]), 0), \
                      np.concatenate((zero_data[train_size_zero:, :-2], one_data[train_size_one:, :-2]), 0)
    y_train, y_test = np.concatenate((zero_data[:train_size_zero, -2].reshape(-1,1), one_data[:train_size_one, -2].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, -2].reshape(-1, 1), one_data[train_size_one:, -2].reshape(-1, 1)), 0)

    model = XGBoostClassifier(lossfunc='LogLoss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_ori = y_pred.copy()
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    result = y_pred - y_test
    print(np.sum(result == 0) / y_pred.shape[0])
    # for i in range(y_test.shape[0]):
    #     print(y_test[i], y_pred[i], y_ori[i])

def main2():
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


    model = XGBoostClassifier(lossfunc='LogLoss', n_estimators=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_ori = y_pred.copy()
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    result = y_pred - y_test
    print(np.sum(result == 0) / y_pred.shape[0])
    for i in range(y_test.shape[0]):
        print(y_test[i], y_pred[i], y_ori[i])

def main3():
    from sklearn.ensemble import GradientBoostingClassifier
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


    model = model = XGBoostClassifier(lossfunc='LogLoss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    result = y_pred - y_test
    print(np.sum(result == 0) / y_pred.shape[0])
    for i in range(y_test.shape[0]):
        print(y_test[i], y_pred[i])

def main6():
    data_train = pd.read_csv('./GiveMeSomeCredit/cs-training.csv')
    data_train = data_train[['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']].values

    data_train.dtype = 'float16'

    data_test = pd.read_csv('./GiveMeSomeCredit/cs-training.csv')
    data_test = data_test[['SeriousDlqin2yrs',
                             'RevolvingUtilizationOfUnsecuredLines', 'age',
                             'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                             'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                             'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                             'NumberOfDependents']].values

    data_test.dtype = 'float16'

    y_train = data_train[:, 0]
    X_train = data_train[:, 1:]
    X_test = data_test[:, 1:]
    model = XGBoostClassifier(lossfunc='LogLoss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_ori = y_pred.copy()
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred2 = y_pred.copy()
    y_pred2[y_pred2 > 0.5] = 1
    y_pred2[y_pred2 <= 0.5] = 0
    print(y_pred)
    # result = y_pred2 - y_test
    # print(np.sum(result == 0) / y_pred.shape[0])

if __name__ == '__main__':
    start = datetime.now()
    main3()
    end = datetime.now()
    print(end - start)


