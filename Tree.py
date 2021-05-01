import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
import math
import time
from SSCalculation import *
from VerticalXGBoost import *
np.random.seed(10)
clientNum = 4
comm = MPI.COMM_WORLD
class Tree:
    def __init__(self, value=None, leftBranch=None, rightBranch=None, col=-1, result=None, isDummy=False):
        self.value = value
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.col = col
        self.result = result
        self.isDummy = isDummy

class VerticalXGBoostTree:
    def __init__(self, rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum):
        self.featureList = []
        self.featureIdxMapping = {}
        self._maxdepth = _maxdepth
        self.rank = rank
        self.loss = lossfunc
        self.split = splitclass
        self._lambda = _lambda / clientNum
        self._gamma = _gamma
        self._epsilon = _epsilon

    def setMapping(self):
        rand = np.random.permutation(self.data.shape[1]) # Get column number
        if self.rank == 0:
            return
        if len(self.featureList) == 0:
            self.featureIdxMapping = {self.featureList[0]:rand[0]}
        else:
            self.featureIdxMapping = {self.featureList[i]:rand[i] for i in range(len(self.featureList))}

    def _split(self, y_and_pred):
        y, y_pred = y_and_pred[:, 0].reshape(-1, 1), y_and_pred[:, 1].reshape(-1, 1)
        return y, y_pred

    def AggBucket(self, shared_G, shared_H):
        bg_Matrix = np.zeros((self.featureNum, self.maxSplitNum))
        bh_Matrix = np.zeros((self.featureNum, self.maxSplitNum))

        for j in range(self.featureNum):
            indexMatrix = np.zeros((self.maxSplitNum, self.data.shape[0]))
            indexMatrixArray = None
            currentRank = None
            if self.rank != 0:
                if j in self.featureList:
                    # print('Rank ' + str(self.rank) + ' I have: ', j)
                    mapped_idx = self.featureIdxMapping[j]
                    splitNum = len(self.quantile[mapped_idx])
                    splitList = self.quantile[mapped_idx]

                    for k in range(splitNum):
                        left = splitList[k][0]
                        right = splitList[k][1]
                        indexMatrix[k, :] = ((self.data[:, self.featureIdxMapping[j]] <= right) & (
                        self.data[:, self.featureIdxMapping[j]] > left)) + 0  # Type conversion
                    indexMatrixArray = self.split.SSSplit(indexMatrix, clientNum)
                    temp = np.zeros_like(indexMatrixArray[0])
                    temp = np.expand_dims(temp, axis=0)
                    indexMatrixArray = np.concatenate([temp, indexMatrixArray], axis=0)
                    comm.send(self.rank, dest=1)
            if self.rank == 1:
                currentRank = comm.recv()
            currentRank = comm.bcast(currentRank, root=1)
            indexMatrix = comm.scatter(indexMatrixArray, root=currentRank)
            bg_Matrix[j, :] = np.sum(self.split.SMUL(indexMatrix, np.tile(shared_G.copy(), (1, self.maxSplitNum)).T, self.rank), axis=1).T
            bh_Matrix[j, :] = np.sum(self.split.SMUL(indexMatrix, np.tile(shared_H.copy(), (1, self.maxSplitNum)).T, self.rank), axis=1).T


        return bg_Matrix, bh_Matrix

    # Implemented SARGMAX, but didn't rip out division.
    def buildTree(self, shared_G, shared_H, shared_S, depth=1):
        shared_gsum = np.sum(shared_G).reshape(-1, 1)
        shared_hsum = np.sum(shared_H).reshape(-1, 1)
        if depth > self._maxdepth:
            a = -shared_gsum
            a = a.reshape(-1, 1)
            b = shared_hsum
            b = b.reshape(-1, 1) + self._lambda
            value = self.split.SDIV(a, b, self.rank)
            return Tree(result=value)
        currentRank = None
        cgain = self.split.SDIV(self.split.SMUL(shared_gsum, shared_gsum, self.rank), shared_hsum + self._lambda, self.rank)


        BG, BH = self.AggBucket(shared_G, shared_H)
        shared_gain = np.zeros((self.featureNum, self.maxSplitNum))
        shared_sl = np.ones((self.data.shape[0], 1))
        shared_sr = np.ones((self.data.shape[0], 1))
        shared_gsum_L = np.array([0.0]).reshape(-1, 1)
        shared_hsum_L = np.array([0.0]).reshape(-1, 1)
        start = None
        if self.rank == 1:
            start = datetime.now()
        for j in range(self.featureNum):
            if self.rank != 0:
                if j in self.featureList:
                    gsum_L, hsum_L = 0, 0
                    gsum_L_array = self.split.SSSplit(np.array([gsum_L]).reshape(-1, 1), clientNum)
                    hsum_L_array = self.split.SSSplit(np.array([hsum_L]).reshape(-1, 1), clientNum)
                    temp = np.zeros_like(gsum_L_array[0])
                    temp = np.expand_dims(temp, axis=0)
                    shared_gsum_L = np.concatenate([temp.copy(), gsum_L_array], axis=0)  # Add zero matrix to rank 0.
                    shared_hsum_L = np.concatenate([temp.copy(), hsum_L_array], axis=0)  # Add zero matrix to rank 0.
                    comm.send(self.rank, dest=1)
            if self.rank == 1:
                currentRank = comm.recv()
            currentRank = comm.bcast(currentRank, root=1)
            shared_gsum_L = comm.scatter(shared_gsum_L, root=currentRank)
            shared_hsum_L = comm.scatter(shared_hsum_L, root=currentRank)

            for k in range(self.maxSplitNum):
                shared_gsum_L += BG[j, k]
                shared_hsum_L += BH[j, k]
                shared_gsum_R = shared_gsum - shared_gsum_L
                shared_hsum_R = shared_hsum - shared_hsum_L
                gain_left = self.split.SDIV(self.split.SMUL(shared_gsum_L, shared_gsum_L, self.rank),
                                             shared_hsum_L + self._lambda, self.rank)
                gain_right = self.split.SDIV(self.split.SMUL(shared_gsum_R, shared_gsum_R, self.rank),
                                             shared_hsum_R + self._lambda, self.rank)
                shared_gain[j, k] = gain_left + gain_right - cgain

        # Combine all the gain from clients, and find the max gain at client 1 (the party who holds label).
        shared_gain /= 2
        shared_gain -= self._gamma / clientNum
        j_best, k_best = self.split.SARGMAX(shared_gain, self.rank)
        if self.rank == 1:
            print(datetime.now() - start)
        gain_sign = self.split.SSIGN(shared_gain[j_best, k_best], self.rank)
        if gain_sign == '+':
            if self.rank != 0:  # Avoid entering calculator node.
                if j_best in self.featureList:
                    sl = np.ones((self.data.shape[0], 1))
                    idx = self.data[:, self.featureIdxMapping[j_best]] > self.quantile[self.featureIdxMapping[j_best]][k_best][1]
                    sl[idx] = 0
                    sr = 1 - sl
                    sl_array = self.split.SSSplit(sl, clientNum)
                    sr_array = self.split.SSSplit(sr, clientNum)
                    temp = np.zeros_like(sl_array[0])
                    temp = np.expand_dims(temp, axis=0)
                    shared_sl = np.concatenate([temp, sl_array], axis=0)  # Add zero matrix to rank 0.
                    shared_sr = np.concatenate([temp, sr_array], axis=0)  # Add zero matrix to rank 0.
                    comm.send(self.rank, dest=1)
            if self.rank == 1:
                currentRank = comm.recv()
            currentRank = comm.bcast(currentRank, root=1)
            shared_sl = comm.scatter(shared_sl, root=currentRank)
            shared_sr = comm.scatter(shared_sr, root=currentRank)

            shared_sl = self.split.SMUL(shared_S, shared_sl, self.rank)
            shared_sr = self.split.SMUL(shared_S, shared_sr, self.rank)
            shared_gl = self.split.SMUL(shared_sl, shared_G, self.rank)
            shared_gr = self.split.SMUL(shared_sr, shared_G, self.rank)
            shared_hl = self.split.SMUL(shared_sl, shared_H, self.rank)
            shared_hr = self.split.SMUL(shared_sr, shared_H, self.rank)

            # print('In build tree, into left', self.rank)
            leftBranch = self.buildTree(shared_gl, shared_hl, shared_sl, depth + 1)
            # print('In build tree, out of left', self.rank)
            rightBranch = self.buildTree(shared_gr, shared_hr, shared_sr, depth + 1)
            # print('In build tree, out of right', self.rank)
            if self.rank != 0:
                if j_best in self.featureList:
                    return Tree(value=self.quantile[self.featureIdxMapping[j_best]][k_best][1], leftBranch=leftBranch,
                                rightBranch=rightBranch, col=j_best, isDummy=False)
                else:
                    return Tree(leftBranch=leftBranch, rightBranch=rightBranch, isDummy=True)  # Return a dummy node
            else:
                return
        else:
            a = -shared_gsum
            a = a.reshape(-1, 1)
            b = shared_hsum
            b = b.reshape(-1, 1) + self._lambda
            value = self.split.SDIV(a, b, self.rank)
            return Tree(result=value)

    # Implemented both SARGMAX and rip out division in LSplit, but calculates leaf weight with SS division.
    def buildTree_ver2(self, shared_G, shared_H, shared_S, depth=1):
        shared_gsum = np.sum(shared_G).reshape(-1, 1)
        shared_hsum = np.sum(shared_H).reshape(-1, 1)
        if depth > self._maxdepth:
            a = -shared_gsum
            a = a.reshape(-1, 1)
            b = shared_hsum
            b = b.reshape(-1, 1) + self._lambda
            value = self.split.SDIV(a, b, self.rank)
            return Tree(result=value)
        currentRank = None
        cgain_up = self.split.SMUL(shared_gsum, shared_gsum, self.rank)
        cgain_down = shared_hsum + self._lambda
        gain_left_up, gain_left_down, gain_right_up, gain_right_down = np.zeros((self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum))
        BG, BH = self.AggBucket(shared_G, shared_H)
        shared_sl = np.ones((self.data.shape[0], 1))
        shared_sr = np.ones((self.data.shape[0], 1))

        for j in range(self.featureNum):
            shared_gsum_L = np.array([0.0]).reshape(-1, 1)
            shared_hsum_L = np.array([0.0]).reshape(-1, 1)

            for k in range(self.maxSplitNum):
                shared_gsum_L += BG[j, k]
                shared_hsum_L += BH[j, k]
                shared_gsum_R = shared_gsum - shared_gsum_L
                shared_hsum_R = shared_hsum - shared_hsum_L
                gain_left_up[j, k] = self.split.SMUL(shared_gsum_L, shared_gsum_L, self.rank)
                gain_left_down[j, k] = shared_hsum_L + self._lambda
                gain_right_up[j, k] = self.split.SMUL(shared_gsum_R, shared_gsum_R, self.rank)
                gain_right_down[j, k] = shared_hsum_R + self._lambda

        # Combine all the gain from clients, and find the max gain at client 1 (the party who holds label).
        j_best, k_best = self.split.SARGMAX_ver2(gain_left_up, gain_left_down, gain_right_up, gain_right_down, self.rank)
        gain_sign = self.split.SSIGN_ver2(gain_left_up[j_best, k_best], gain_left_down[j_best, k_best], gain_right_up[j_best, k_best], gain_right_down[j_best, k_best], cgain_up, cgain_down, self._gamma, self.rank)
        if gain_sign == '+':
            if self.rank != 0:  # Avoid entering calculator node.
                if j_best in self.featureList:
                    sl = np.ones((self.data.shape[0], 1))
                    idx = self.data[:, self.featureIdxMapping[j_best]] > \
                              self.quantile[self.featureIdxMapping[j_best]][k_best][1]
                    sl[idx] = 0
                    sr = 1 - sl
                    sl_array = self.split.SSSplit(sl, clientNum)
                    sr_array = self.split.SSSplit(sr, clientNum)
                    temp = np.zeros_like(sl_array[0])
                    temp = np.expand_dims(temp, axis=0)
                    shared_sl = np.concatenate([temp, sl_array], axis=0)  # Add zero matrix to rank 0.
                    shared_sr = np.concatenate([temp, sr_array], axis=0)  # Add zero matrix to rank 0.
                    comm.send(self.rank, dest=1)
            if self.rank == 1:
                currentRank = comm.recv()
            currentRank = comm.bcast(currentRank, root=1)
            shared_sl = comm.scatter(shared_sl, root=currentRank)
            shared_sr = comm.scatter(shared_sr, root=currentRank)

            shared_sl = self.split.SMUL(shared_S, shared_sl, self.rank)
            shared_sr = self.split.SMUL(shared_S, shared_sr, self.rank)
            shared_gl = self.split.SMUL(shared_sl, shared_G, self.rank)
            shared_gr = self.split.SMUL(shared_sr, shared_G, self.rank)
            shared_hl = self.split.SMUL(shared_sl, shared_H, self.rank)
            shared_hr = self.split.SMUL(shared_sr, shared_H, self.rank)

            leftBranch = self.buildTree_ver2(shared_gl, shared_hl, shared_sl, depth + 1)
            rightBranch = self.buildTree_ver2(shared_gr, shared_hr, shared_sr, depth + 1)
            if self.rank != 0:
                if j_best in self.featureList:
                    print(depth, rank)
                    return Tree(value=self.quantile[self.featureIdxMapping[j_best]][k_best][1],
                                    leftBranch=leftBranch,
                                    rightBranch=rightBranch, col=j_best, isDummy=False)
                else:
                    print(depth, 'None', rank)
                    return Tree(leftBranch=leftBranch, rightBranch=rightBranch, isDummy=True)  # Return a dummy node
            else:
                return
        else:
            a = -shared_gsum
            a = a.reshape(-1, 1)
            b = shared_hsum
            b = b.reshape(-1, 1) + self._lambda
            value = self.split.SDIV(a, b, self.rank)
            return Tree(result=value)

    # Implement the Fisrt-tree trick from Kewei Cheng's paper, but calculates leaf weight with SS division.
    def buildTree_ver3(self, shared_G, shared_H, shared_S, depth=1, tree_num=0):
        shared_gsum = np.sum(shared_G).reshape(-1, 1)
        shared_hsum = np.sum(shared_H).reshape(-1, 1)
        if depth > self._maxdepth:
            a = -shared_gsum
            a = a.reshape(-1, 1)
            b = shared_hsum
            b = b.reshape(-1, 1) + self._lambda
            value = self.split.SDIV(a, b, self.rank)
            return Tree(result=value)
        currentRank = None
        cgain_up = self.split.SMUL(shared_gsum, shared_gsum, self.rank)
        cgain_down = shared_hsum + self._lambda
        gain_left_up, gain_left_down, gain_right_up, gain_right_down = np.zeros(
            (self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum)), np.zeros(
            (self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum))
        BG, BH = self.AggBucket(shared_G, shared_H)
        shared_sl = np.ones((self.data.shape[0], 1))
        shared_sr = np.ones((self.data.shape[0], 1))

        for j in range(self.featureNum):
            if tree_num == 0: # The first tree.
                if self.rank == 1: # The first party who holds labels.
                    if j not in self.featureList:
                        permission = False
                    else:
                        permission = True
                    comm.send(permission, dest=0)
                    for i in range(2, clientNum + 1):
                        comm.send(permission, dest=i)
                else:
                    permission = comm.recv(source=1)
                if not permission:
                    continue  # Jump to the next feature
            shared_gsum_L = np.array([0.0]).reshape(-1, 1)
            shared_hsum_L = np.array([0.0]).reshape(-1, 1)
            for k in range(self.maxSplitNum):
                shared_gsum_L += BG[j, k]
                shared_hsum_L += BH[j, k]
                shared_gsum_R = shared_gsum - shared_gsum_L
                shared_hsum_R = shared_hsum - shared_hsum_L
                gain_left_up[j, k] = self.split.SMUL(shared_gsum_L, shared_gsum_L, self.rank)
                gain_left_down[j, k] = shared_hsum_L + self._lambda
                gain_right_up[j, k] = self.split.SMUL(shared_gsum_R, shared_gsum_R, self.rank)
                gain_right_down[j, k] = shared_hsum_R + self._lambda


        # Combine all the gain from clients, and find the max gain at client 1 (the party who holds label).
        j_best, k_best = self.split.SARGMAX_ver3(gain_left_up, gain_left_down, gain_right_up, gain_right_down,
                                                 self.rank, tree_num, self.featureList)
        gain_sign = self.split.SSIGN_ver2(gain_left_up[j_best, k_best], gain_left_down[j_best, k_best],
                                          gain_right_up[j_best, k_best], gain_right_down[j_best, k_best],
                                          cgain_up, cgain_down, self._gamma, self.rank)
        if gain_sign == '+':
            if self.rank != 0:  # Avoid entering calculator node.
                if j_best in self.featureList:
                    sl = np.ones((self.data.shape[0], 1))
                    idx = self.data[:, self.featureIdxMapping[j_best]] > \
                          self.quantile[self.featureIdxMapping[j_best]][k_best][1]
                    sl[idx] = 0
                    sr = 1 - sl
                    sl_array = self.split.SSSplit(sl, clientNum)
                    sr_array = self.split.SSSplit(sr, clientNum)
                    temp = np.zeros_like(sl_array[0])
                    temp = np.expand_dims(temp, axis=0)
                    shared_sl = np.concatenate([temp, sl_array], axis=0)  # Add zero matrix to rank 0.
                    shared_sr = np.concatenate([temp, sr_array], axis=0)  # Add zero matrix to rank 0.
                    comm.send(self.rank, dest=0)
            if self.rank == 0:
                currentRank = comm.recv()
            currentRank = comm.bcast(currentRank, root=0)
            shared_sl = comm.scatter(shared_sl, root=currentRank)
            shared_sr = comm.scatter(shared_sr, root=currentRank)

            shared_sl = self.split.SMUL(shared_S, shared_sl, self.rank)
            shared_sr = self.split.SMUL(shared_S, shared_sr, self.rank)
            shared_gl = self.split.SMUL(shared_sl, shared_G, self.rank)
            shared_gr = self.split.SMUL(shared_sr, shared_G, self.rank)
            shared_hl = self.split.SMUL(shared_sl, shared_H, self.rank)
            shared_hr = self.split.SMUL(shared_sr, shared_H, self.rank)

            leftBranch = self.buildTree_ver3(shared_gl, shared_hl, shared_sl, depth + 1, tree_num)
            rightBranch = self.buildTree_ver3(shared_gr, shared_hr, shared_sr, depth + 1, tree_num)
            if self.rank != 0:
                if j_best in self.featureList:
                    return Tree(value=self.quantile[self.featureIdxMapping[j_best]][k_best][1],
                                leftBranch=leftBranch,
                                rightBranch=rightBranch, col=j_best, isDummy=False)
                else:
                    return Tree(leftBranch=leftBranch, rightBranch=rightBranch,
                                isDummy=True)  # Return a dummy node
            else:
                return
        else:
            a = -shared_gsum
            a = a.reshape(-1, 1)
            b = shared_hsum
            b = b.reshape(-1, 1) + self._lambda
            value = self.split.SDIV(a, b, self.rank)
            return Tree(result=value)

    # Implement the first-layer mask and gradient descent.
    def buildTree_ver4(self, shared_G, shared_H, shared_S, depth=1):
        shared_gsum = np.sum(shared_G).reshape(-1, 1)
        shared_hsum = np.sum(shared_H).reshape(-1, 1)
        iter = 10
        if depth > self._maxdepth:
            a = shared_hsum
            a = a.reshape(-1, 1) + self._lambda
            a *= 0.5
            b = shared_gsum
            b = b.reshape(-1, 1)
            value = self.split.S_GD(a, b, self.rank, lamb=self._lambda)
            return Tree(result=value)

        currentRank = None
        cgain_up = self.split.SMUL(shared_gsum, shared_gsum, self.rank)
        cgain_down = shared_hsum + self._lambda
        gain_left_up, gain_left_down, gain_right_up, gain_right_down = np.zeros(
            (self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum)), np.zeros(
            (self.featureNum, self.maxSplitNum)), np.zeros((self.featureNum, self.maxSplitNum))
        BG, BH = self.AggBucket(shared_G, shared_H)
        shared_sl = np.ones((self.data.shape[0], 1))
        shared_sr = np.ones((self.data.shape[0], 1))

        start = None
        if self.rank == 1:
            start = datetime.now()
        for j in range(self.featureNum):
            shared_gsum_L = np.array([np.sum(BG[j, :k+1]) for k in range(self.maxSplitNum)])
            shared_hsum_L = np.array([np.sum(BH[j, :k+1]) for k in range(self.maxSplitNum)])
            shared_gsum_R = (shared_gsum - shared_gsum_L).reshape(-1,)
            shared_hsum_R = (shared_hsum - shared_hsum_L).reshape(-1,)
            gain_left_up[j, :] = self.split.SMUL(shared_gsum_L, shared_gsum_L, self.rank).T
            gain_left_down[j, :] = shared_hsum_L + self._lambda
            gain_right_up[j, :] = self.split.SMUL(shared_gsum_R, shared_gsum_R, self.rank).T
            gain_right_down[j, :] = shared_hsum_R + self._lambda

        # First-Layer-Mask
        j_best, k_best = self.split.SARGMAX_ver4(gain_left_up, gain_left_down, gain_right_up, gain_right_down,
                                                 self.rank, depth, self.featureList)

        # Original
        # j_best, k_best = self.split.SARGMAX_ver2(gain_left_up, gain_left_down, gain_right_up, gain_right_down,
        #                                          self.rank)

        if self.rank == 1:
            print(datetime.now() - start)
        gain_sign = self.split.SSIGN_ver2(gain_left_up[j_best, k_best], gain_left_down[j_best, k_best],
                                          gain_right_up[j_best, k_best], gain_right_down[j_best, k_best],
                                          cgain_up, cgain_down, self._gamma, self.rank)
        if gain_sign == '+' or depth == 1:  # For layer 1, splitte by the first party, we should pass it.
            if self.rank != 0:  # Avoid entering calculator node.
                if j_best in self.featureList:
                    sl = np.ones((self.data.shape[0], 1))
                    idx = self.data[:, self.featureIdxMapping[j_best]] > \
                          self.quantile[self.featureIdxMapping[j_best]][k_best][1]
                    sl[idx] = 0
                    sr = 1 - sl
                    sl_array = self.split.SSSplit(sl, clientNum)
                    sr_array = self.split.SSSplit(sr, clientNum)
                    temp = np.zeros_like(sl_array[0])
                    temp = np.expand_dims(temp, axis=0)
                    shared_sl = np.concatenate([temp, sl_array], axis=0)  # Add zero matrix to rank 0.
                    shared_sr = np.concatenate([temp, sr_array], axis=0)  # Add zero matrix to rank 0.
                    comm.send(self.rank, dest=1)
            if self.rank == 1:
                currentRank = comm.recv()
            currentRank = comm.bcast(currentRank, root=1)
            shared_sl = comm.scatter(shared_sl, root=currentRank)
            shared_sr = comm.scatter(shared_sr, root=currentRank)

            shared_sl = self.split.SMUL(shared_S, shared_sl, self.rank)
            shared_sr = self.split.SMUL(shared_S, shared_sr, self.rank)
            shared_gl = self.split.SMUL(shared_sl, shared_G, self.rank)
            shared_gr = self.split.SMUL(shared_sr, shared_G, self.rank)
            shared_hl = self.split.SMUL(shared_sl, shared_H, self.rank)
            shared_hr = self.split.SMUL(shared_sr, shared_H, self.rank)

            leftBranch = self.buildTree_ver4(shared_gl, shared_hl, shared_sl, depth + 1)
            rightBranch = self.buildTree_ver4(shared_gr, shared_hr, shared_sr, depth + 1)
            if self.rank != 0:
                if j_best in self.featureList:
                    return Tree(value=self.quantile[self.featureIdxMapping[j_best]][k_best][1],
                                leftBranch=leftBranch,
                                rightBranch=rightBranch, col=j_best, isDummy=False)
                else:
                    return Tree(leftBranch=leftBranch, rightBranch=rightBranch,
                                isDummy=True)  # Return a dummy node
            else:
                return

        else:
            a = shared_hsum
            a = a.reshape(-1, 1) + self._lambda
            a *= 0.5
            b = shared_gsum
            b = b.reshape(-1, 1)
            value = self.split.S_GD(a, b, self.rank, lamb=self._lambda)
            return Tree(result=value)

    # This function contains no communication operation.
    def getInfo(self, tree, data, belongs=1):
        if self.rank == 0:
            return
        if tree.result != None:
            return np.array([belongs]).reshape(-1, 1), np.array([tree.result]).reshape(-1, 1)
        else:
            left_belongs = 0
            right_belongs = 0
            if tree.isDummy:
                if belongs == 1:
                    left_belongs = 1
                    right_belongs = 1
                left_idx, left_result = self.getInfo(tree.leftBranch, data, left_belongs)
                right_idx, right_result = self.getInfo(tree.rightBranch, data, right_belongs)
                idx = np.concatenate((left_idx, right_idx), axis=0)
                result = np.concatenate((left_result, right_result), axis=0)
                return idx, result

            v = data[0, self.featureIdxMapping[tree.col]]

            if belongs == 1: # In selected branch
                if v > tree.value:
                    right_belongs = 1
                else:
                    left_belongs = 1
            left_idx, left_result = self.getInfo(tree.leftBranch, data, left_belongs)
            right_idx, right_result = self.getInfo(tree.rightBranch, data, right_belongs)
            idx = np.concatenate((left_idx, right_idx), axis=0)
            result = np.concatenate((left_result, right_result), axis=0)
            return idx, result

    def fit(self, y_and_pred, tree_num):
        size = None
        size_list = comm.gather(self.data.shape[1], root=2)  # Gather all the feature size.
        if self.rank == 2:
            size = sum(size_list[1:])
        self.featureNum = comm.bcast(size, root=2)  # Broadcast how many feature there are in total.
        if self.rank == 2:
            random_list = np.random.permutation(self.featureNum)
            start = 0
            for i in range(1, clientNum + 1):
                rand = random_list[start:start + size_list[i]]
                if i == 2:
                    self.featureList = rand
                else:
                    comm.send(rand, dest=i)  # Send random_list to all the client, mask their feature index.
                start += size_list[i]
        elif self.rank != 0:
            self.featureList = comm.recv(source=2)
        self.setMapping()
        shared_G, shared_H, shared_S = None, None, None
        if self.rank == 1: # Calculate gradients on the node who have labels.
            y, y_pred = self._split(y_and_pred)
            G = self.loss.gradient(y, y_pred)
            H = self.loss.hess(y, y_pred)
            S = np.ones_like(y)
            shared_G = self.split.SSSplit(G, clientNum) # Split G/H/indicator.
            shared_H = self.split.SSSplit(H, clientNum)
            shared_S = self.split.SSSplit(S, clientNum)
            temp = np.zeros_like(shared_G[0])
            temp = np.expand_dims(temp, axis=0)
            shared_G = np.concatenate([temp.copy(), shared_G], axis=0)
            shared_H = np.concatenate([temp.copy(), shared_H], axis=0)
            shared_S = np.concatenate([temp.copy(), shared_S], axis=0)

        shared_G = comm.scatter(shared_G, root=1)
        shared_H = comm.scatter(shared_H, root=1)
        shared_S = comm.scatter(shared_S, root=1)


        self.Tree = self.buildTree_ver4(shared_G, shared_H, shared_S)
        # self.Tree = self.buildTree_ver3(shared_G, shared_H, shared_S, depth=1, tree_num=tree_num)
        # self.Tree = self.buildTree_ver2(shared_G, shared_H, shared_S)
        # self.Tree = self.buildTree(shared_G, shared_H, shared_S)

    def classify(self, tree, data):
        idx_list = []
        shared_idx = None
        final_result = 0
        if self.rank != 0:
            idx, result = self.getInfo(tree, data)
        for i in range(1, clientNum + 1):
            if self.rank == i:
                shared_idx = self.split.SSSplit(idx, clientNum)
                temp = np.zeros_like(shared_idx[0])
                temp = np.expand_dims(temp, axis=0)
                shared_idx = np.concatenate([temp, shared_idx], axis=0)
            shared_idx = comm.scatter(shared_idx, root=i)
            idx_list.append(shared_idx)

        final_idx = idx_list[0]
        for i in range(1, clientNum):
            final_idx = self.split.SMUL(final_idx, idx_list[i], self.rank)
        if self.rank == 0:
            result = np.zeros_like(final_idx)
        temp_result = np.sum(self.split.SMUL(final_idx, result, self.rank))
        temp_result = comm.gather(temp_result, root=1)
        if self.rank == 1:
            final_result = np.sum(temp_result[1:])
        return final_result

    def predict(self, data): # Encapsulated for many data
        data_num = data.shape[0]
        result = []
        for i in range(data_num):
            temp_result = self.classify(self.Tree, data[i].reshape(1, -1))
            if self.rank == 1:
                result.append(temp_result)
            else:
                pass
        result = np.array(result).reshape((-1, 1))
        return result