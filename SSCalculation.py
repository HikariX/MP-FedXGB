import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
import math
import time
from VerticalXGBoost import *
from Tree import *
np.random.seed(10)
clientNum = 4
comm = MPI.COMM_WORLD
class SSCalculate:
    def SSSplit(self, data, clientNum):
        r = np.array([np.random.uniform(0, 4, (data.shape[0], data.shape[1])) for i in range(clientNum - 1)])
        data = data.astype('float64')
        data -= np.sum(r, axis=0).astype('float64')
        data = np.expand_dims(data, axis=0)
        dataList = np.concatenate([r, data], axis=0)
        return dataList

    def SMUL(self, data_A, data_B, rank):
        if len(data_A.shape) <= 1:
            data_A = data_A.reshape(-1, 1)
            data_B = data_B.reshape(-1, 1)
        if rank == 0: # Send shared data
            a = np.random.rand(data_A.shape[0], data_A.shape[1])
            b = np.random.rand(data_A.shape[0], data_A.shape[1])
            c = a * b
            dataList_a = self.SSSplit(a, clientNum)
            dataList_b = self.SSSplit(b, clientNum)
            dataList_c = self.SSSplit(c, clientNum)
            for i in range(1, clientNum + 1):
                comm.send([dataList_a[i - 1], dataList_b[i - 1], dataList_c[i - 1]], dest=i)
            return a
        elif rank == 1:
            ra, rb, rc = comm.recv(source=0)
            ei = data_A - ra
            fi = data_B - rb
            eList = []
            fList = []
            for i in range(2, clientNum + 1):
                temp_e, temp_f = comm.recv(source=i)
                eList.append(temp_e)
                fList.append(temp_f)

            e = np.sum(np.array(eList), axis=0) + ei
            f = np.sum(np.array(fList), axis=0) + fi
            for i in range(2, clientNum + 1):
                comm.send((e, f), dest=i)
            zi = e * f + f * ra + e * rb + rc
            return zi
        else:
            ra, rb, rc = comm.recv(source=0)
            ei = data_A - ra
            fi = data_B - rb
            comm.send((ei, fi), dest=1)
            e, f = comm.recv(source=1)
            zi = f * ra + e * rb + rc
            return zi

    def SDIV(self, data_A, data_B, rank):
        # iter = 8
        # factor = 1.9
        if len(data_A.shape) <= 1:
            data_A = data_A.reshape(-1, 1)
            data_B = data_B.reshape(-1, 1)
        iter = 20
        if rank == 0: # Send shared data
            divisor_list = []
            for i in range(1, clientNum + 1):
                divisor_list.append(comm.recv(source=i))
            divisor_list = np.array(divisor_list)
            divisor = np.min(divisor_list, axis=0) / 10
            divisor /= clientNum # Equally share the divisor to parties.
            # divisor = divisor / np.ceil(clientNum / 2)
            for i in range(1, clientNum + 1):
                comm.send(divisor, dest=i)
            for i in range(iter):
                self.SMUL(data_B, divisor, rank)
                self.SMUL(divisor, divisor, rank)
            self.SMUL(data_A, data_B, rank)
            return divisor
        else:
            divisor = np.zeros_like(data_B)
            divisor.dtype = np.float64
            for i in range(data_B.shape[0]):
                for j in range(data_B.shape[1]):
                    k = 0
                    data = abs(data_B[i, j])
                    if data > 1:
                        while data >= 1:
                            data /= 10
                            k += 1
                        divisor[i, j] = 1 / pow(10, k)
                    else:
                        while data <= 1:
                            data *= 10
                            k += 1
                        k -= 1
                        divisor[i, j] = 1 * pow(10, k)
            comm.send(divisor, dest=0)
            divisor = comm.recv(source=0)
            for i in range(iter):
                t = 2 / clientNum - self.SMUL(data_B, divisor, rank)
                divisor_next = self.SMUL(divisor, t, rank)
                divisor = divisor_next
            result = self.SMUL(data_A, divisor, rank)
            return result

    # Implement ARGMAX by calculating SS division in build_tree.
    def SARGMAX(self, data, rank):
        row_idx = None
        col_idx = None
        total_value_list = None
        sign_list = None
        col_index_list = None
        new_col_index_list = None
        new_row_index_list = None
        row_index_list = None
        row_idx_dict = {}
        for k in range(data.shape[0]):
            ori_value_list = data[k, :]
            value_list = ori_value_list.copy()
            col_index_list = [i for i in range(0, len(ori_value_list))]
            while ori_value_list.shape[0] > 1:
                if rank != 0:
                    if len(ori_value_list) % 2 == 0: # Even
                        value_list = [ori_value_list[i] - ori_value_list[i + 1] for i in range(0, len(ori_value_list), 2)]
                    else:
                        value_list = [ori_value_list[i] - ori_value_list[i + 1] for i in range(0, len(ori_value_list) - 1, 2)]
                        value_list.append(value_list[-1])
                    value_list = np.array(value_list)
                total_value_list = comm.gather(value_list, root=0)
                if rank == 0:
                    total_value_list = total_value_list[1:] # Rip out the nonsense list from rank 0.
                    shared_value_sum = np.sum(np.array(total_value_list), axis=0)
                    sign_list = np.array(shared_value_sum >= 0) # Record the judgement.
                    new_col_index_list = []
                    iter_size = len(sign_list)
                    if len(ori_value_list) % 2 != 0:
                        iter_size -= 1
                    for j in range(iter_size):
                        if sign_list[j]: # True, or the former value is bigger than the latter.
                            new_col_index_list.append(col_index_list[j * 2])
                        else:
                            new_col_index_list.append(col_index_list[j * 2 + 1])
                    if len(ori_value_list) % 2 != 0: # Odd
                        new_col_index_list.append(col_index_list[-1])
                new_col_index_list = comm.bcast(new_col_index_list, root=0)
                ori_value_list = np.array([data[k, i] for i in new_col_index_list])
                col_index_list = new_col_index_list
            col_idx = col_index_list[0] # Retrieve out the only col index.
            row_idx_dict[k] = col_idx

        ori_value_list = np.array([data[i, row_idx_dict[i]] for i in row_idx_dict.keys()])
        value_list = ori_value_list.copy()
        row_index_list = [i for i in range(0, len(ori_value_list))]
        while ori_value_list.shape[0] > 1:
            if rank != 0:
                if len(ori_value_list) % 2 == 0: # Even
                    value_list = [ori_value_list[i] - ori_value_list[i + 1] for i in range(0, len(ori_value_list), 2)]
                else:
                    value_list = [ori_value_list[i] - ori_value_list[i + 1] for i in range(0, len(ori_value_list) - 1, 2)]
                    value_list.append(ori_value_list[-1])
                value_list = np.array(value_list)
            total_value_list = comm.gather(value_list, root=0)
            if rank == 0:
                total_value_list = total_value_list[1:] # Rip out the nonsense list from rank 0.
                shared_value_sum = np.sum(np.array(total_value_list), axis=0)
                sign_list = np.array(shared_value_sum >= 0) # Record the judgement.
                new_row_index_list = []
                iter_size = len(sign_list)
                if len(ori_value_list) % 2 != 0:
                    iter_size -= 1
                for j in range(iter_size):
                    if sign_list[j]: # True, or the former value is bigger than the latter.
                        new_row_index_list.append(row_index_list[j * 2])
                    else:
                        new_row_index_list.append(row_index_list[j * 2 + 1])
                if len(ori_value_list) % 2 != 0: # Odd
                    new_row_index_list.append(row_index_list[-1])
            new_row_index_list = comm.bcast(new_row_index_list, root=0)
            ori_value_list = np.array([data[i, row_idx_dict[i]] for i in new_row_index_list])
            row_index_list = new_row_index_list
        return row_index_list[0], row_idx_dict[row_index_list[0]] # Return feature and split position

    # Implement ARGMAX and rip out SS division.
    def SARGMAX_ver2(self, gain_left_up, gain_left_down, gain_right_up, gain_right_down, rank):
        new_col_index_list = None
        new_row_index_list = None
        row_idx_dict = {}
        row_num = gain_left_up.shape[0]
        nominator_sign_list = denominator_sign_list = None
        for k in range(row_num):
            col_index_list = [i for i in range(0, len(gain_right_down[0, :]))]
            while len(col_index_list) > 1:
                iter_size = len(col_index_list)
                if iter_size % 2 != 0:  # Odd
                    iter_size -= 1
                list1 = [gain_left_up[k, col_index_list[i]] for i in range(0, iter_size, 2)]
                list2 = [gain_right_down[k, col_index_list[i]] for i in range(0, iter_size, 2)]
                list3 = [gain_right_up[k, col_index_list[i]] for i in range(0, iter_size, 2)]
                list4 = [gain_left_down[k, col_index_list[i]] for i in range(0, iter_size, 2)]

                list5 = [gain_left_up[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list6 = [gain_right_down[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list7 = [gain_right_up[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list8 = [gain_left_down[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]

                nominator1 = self.SMUL(np.array(list1), np.array(list8), rank) - self.SMUL(np.array(list5),
                                                                                           np.array(list4), rank)
                nominator2 = self.SMUL(np.array(list3), np.array(list6), rank) - self.SMUL(np.array(list7),
                                                                                           np.array(list2), rank)
                denominator1 = self.SMUL(np.array(list4), np.array(list8), rank)
                denominator2 = self.SMUL(np.array(list2), np.array(list6), rank)

                total_nominator = self.SMUL(nominator1, denominator2, rank) + self.SMUL(nominator2, denominator1, rank)
                total_denominator = self.SMUL(denominator1, denominator2, rank)

                total_nominator_list = comm.gather(total_nominator, root=2)
                total_deominator_list = comm.gather(total_denominator, root=1)

                if rank == 2:
                    total_nominator_list = total_nominator_list[1:]
                    nominator_sign_list = np.sum(np.array(total_nominator_list), axis=0)
                    nominator_sign_list[nominator_sign_list >= 0] = 1
                    nominator_sign_list[nominator_sign_list < 0] = -1
                    comm.send(nominator_sign_list, dest=1)
                elif rank == 1:
                    total_denominator_list = total_deominator_list[1:]
                    denominator_sign_list = np.sum(np.array(total_denominator_list), axis=0)
                    denominator_sign_list[denominator_sign_list >= 0] = 1
                    denominator_sign_list[denominator_sign_list < 0] = -1
                    nominator_sign_list = comm.recv(source=2)
                    sign_list = denominator_sign_list * nominator_sign_list # Record the judgement.
                    sign_list = sign_list >= 0 + 0
                    new_col_index_list = []
                    iter_size = len(sign_list)
                    for j in range(iter_size):
                        if sign_list[j]:  # True, or the former value is bigger than the latter.
                            new_col_index_list.append(col_index_list[j * 2])
                        else:
                            new_col_index_list.append(col_index_list[j * 2 + 1])
                    if len(col_index_list) % 2 != 0:  # Odd
                        new_col_index_list.append(col_index_list[-1])
                new_col_index_list = comm.bcast(new_col_index_list, root=1)
                col_index_list = new_col_index_list
            col_idx = col_index_list[0]  # Retrieve out the only col index.
            row_idx_dict[k] = col_idx

        row_index_list = [i for i in row_idx_dict.keys()]
        nominator_sign_list = denominator_sign_list = None
        while len(row_index_list) > 1:
            iter_size = len(row_index_list)
            if len(row_index_list) % 2 != 0:  # Odd
                iter_size -= 1
            list1 = [gain_left_up[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list2 = [gain_right_down[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list3 = [gain_right_up[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list4 = [gain_left_down[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list5 = [gain_left_up[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list6 = [gain_right_down[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list7 = [gain_right_up[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list8 = [gain_left_down[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            nominator1 = self.SMUL(np.array(list1), np.array(list8), rank) - self.SMUL(np.array(list5),
                                                                                       np.array(list4),
                                                                                       rank)
            nominator2 = self.SMUL(np.array(list3), np.array(list6), rank) - self.SMUL(np.array(list7),
                                                                                       np.array(list2),
                                                                                       rank)
            denominator1 = self.SMUL(np.array(list4), np.array(list8), rank)
            denominator2 = self.SMUL(np.array(list2), np.array(list6), rank)

            total_nominator = self.SMUL(nominator1, denominator2, rank) + self.SMUL(nominator2, denominator1, rank)
            total_denominator = self.SMUL(denominator1, denominator2, rank)

            total_nominator_list = comm.gather(total_nominator, root=2)
            total_deominator_list = comm.gather(total_denominator, root=1)

            if rank == 2:
                total_nominator_list = total_nominator_list[1:]
                nominator_sign_list = np.sum(np.array(total_nominator_list), axis=0)
                nominator_sign_list[nominator_sign_list >= 0] = 1
                nominator_sign_list[nominator_sign_list < 0] = -1
                comm.send(nominator_sign_list, dest=1)
            elif rank == 1:
                total_denominator_list = total_deominator_list[1:]
                denominator_sign_list = np.sum(np.array(total_denominator_list), axis=0)
                denominator_sign_list[denominator_sign_list >= 0] = 1
                denominator_sign_list[denominator_sign_list < 0] = -1
                nominator_sign_list = comm.recv(source=2)
                sign_list = denominator_sign_list * nominator_sign_list  # Record the judgement.
                sign_list = sign_list >= 0 + 0
                new_row_index_list = []
                iter_size = len(sign_list)
                for j in range(iter_size):
                    if sign_list[j]:  # True, or the former value is bigger than the latter.
                        new_row_index_list.append(row_index_list[j * 2])
                    else:
                        new_row_index_list.append(row_index_list[j * 2 + 1])
                if len(row_index_list) % 2 != 0:  # Odd
                    new_row_index_list.append(row_index_list[-1])
            new_row_index_list = comm.bcast(new_row_index_list, root=1)
            row_index_list = new_row_index_list
        # if rank == 0:
        #     print(row_index_list[0], row_idx_dict[row_index_list[0]])
        return row_index_list[0], row_idx_dict[row_index_list[0]]  # Return feature and split position

    # Implement the Fisrt-tree trick from Kewei Cheng's paper.
    def SARGMAX_ver3(self, gain_left_up, gain_left_down, gain_right_up, gain_right_down, rank, tree_num, legal_featureList):
        new_col_index_list = None
        new_row_index_list = None
        row_idx_dict = {}
        row_num = gain_left_up.shape[0]
        permission = True
        for k in range(row_num):
            if tree_num == 0: # The first tree.
                if rank == 1: # The first party who holds labels.
                    if k not in legal_featureList:
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
            gain_left_up_ori = gain_left_up[k, :]
            gain_left_down_ori = gain_left_down[k, :]
            gain_right_up_ori = gain_right_up[k, :]
            gain_right_down_ori = gain_right_down[k, :]
            value_list = np.zeros_like(gain_left_up_ori)
            col_index_list = [i for i in range(0, len(value_list))]
            while len(col_index_list) > 1:
                iter_size = len(col_index_list)
                if len(col_index_list) % 2 != 0: # Odd
                    iter_size -= 1
                list1 = [gain_left_up_ori[col_index_list[i]] for i in range(0, iter_size, 2)]
                list2 = [gain_right_down_ori[col_index_list[i]] for i in range(0, iter_size, 2)]
                list3 = [gain_right_up_ori[col_index_list[i]] for i in range(0, iter_size, 2)]
                list4 = [gain_left_down_ori[col_index_list[i]] for i in range(0, iter_size, 2)]

                list5 = [gain_left_up_ori[col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list6 = [gain_right_down_ori[col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list7 = [gain_right_up_ori[col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list8 = [gain_left_down_ori[col_index_list[i + 1]] for i in range(0, iter_size, 2)]

                nominator1 = self.SMUL(np.array(list1), np.array(list8), rank) - self.SMUL(np.array(list5), np.array(list4), rank)
                nominator2 = self.SMUL(np.array(list3), np.array(list6), rank) - self.SMUL(np.array(list7), np.array(list2), rank)
                denominator1 = self.SMUL(np.array(list4), np.array(list8), rank)
                denominator2 = self.SMUL(np.array(list2), np.array(list6), rank)

                total_nominator1_list = comm.gather(nominator1, root=2)
                total_nominator2_list = comm.gather(nominator2, root=2)

                total_denominator1_list = comm.gather(denominator1, root=2)
                total_denominator2_list = comm.gather(denominator2, root=2)
                if rank == 2:
                    total_nominator1_list = total_nominator1_list[1:] # Rip out the nonsense list from rank 0.
                    total_nominator2_list = total_nominator2_list[1:]
                    total_denominator1_list = total_denominator1_list[1:]
                    total_denominator2_list = total_denominator2_list[1:]

                    shared_nominator1_sum = np.sum(np.array(total_nominator1_list), axis=0)
                    shared_nominator2_sum = np.sum(np.array(total_nominator2_list), axis=0)
                    shared_denominator1_sum = np.sum(np.array(total_denominator1_list), axis=0)
                    shared_denominator2_sum = np.sum(np.array(total_denominator2_list), axis=0)
                    shared_value_final = shared_nominator1_sum / shared_denominator1_sum + shared_nominator2_sum / shared_denominator2_sum
                    sign_list = np.array(shared_value_final >= 0) # Record the judgement.
                    new_col_index_list = []
                    iter_size = len(sign_list)
                    for j in range(iter_size):
                        if sign_list[j]: # True, or the former value is bigger than the latter.
                            new_col_index_list.append(col_index_list[j * 2])
                        else:
                            new_col_index_list.append(col_index_list[j * 2 + 1])
                    if len(col_index_list) % 2 != 0: # Odd
                        new_col_index_list.append(col_index_list[-1])
                new_col_index_list = comm.bcast(new_col_index_list, root=2)
                col_index_list = new_col_index_list
            col_idx = col_index_list[0] # Retrieve out the only col index.
            row_idx_dict[k] = col_idx

        row_index_list = [i for i in row_idx_dict.keys()]
        while len(row_index_list) > 1:
            iter_size = len(row_index_list)
            if len(row_index_list) % 2 != 0:  # Odd
                iter_size -= 1
            list1 = [gain_left_up[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in range(0, iter_size, 2)]
            list2 = [gain_right_down[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list3 = [gain_right_up[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in range(0, iter_size, 2)]
            list4 = [gain_left_down[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in range(0, iter_size, 2)]
            list5 = [gain_left_up[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list6 = [gain_right_down[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list7 = [gain_right_up[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list8 = [gain_left_down[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            nominator1 = self.SMUL(np.array(list1), np.array(list8), rank) - self.SMUL(np.array(list5), np.array(list4),
                                                                                       rank)
            nominator2 = self.SMUL(np.array(list3), np.array(list6), rank) - self.SMUL(np.array(list7), np.array(list2),
                                                                                       rank)
            denominator1 = self.SMUL(np.array(list4), np.array(list8), rank)
            denominator2 = self.SMUL(np.array(list2), np.array(list6), rank)

            total_nominator1_list = comm.gather(nominator1, root=2)
            total_nominator2_list = comm.gather(nominator2, root=2)

            total_denominator1_list = comm.gather(denominator1, root=2)
            total_denominator2_list = comm.gather(denominator2, root=2)
            if rank == 2:
                total_nominator1_list = total_nominator1_list[1:]  # Rip out the nonsense list from rank 0.
                total_nominator2_list = total_nominator2_list[1:]
                total_denominator1_list = total_denominator1_list[1:]
                total_denominator2_list = total_denominator2_list[1:]

                shared_nominator1_sum = np.sum(np.array(total_nominator1_list), axis=0)
                shared_nominator2_sum = np.sum(np.array(total_nominator2_list), axis=0)
                shared_denominator1_sum = np.sum(np.array(total_denominator1_list), axis=0)
                shared_denominator2_sum = np.sum(np.array(total_denominator2_list), axis=0)
                shared_value_final = shared_nominator1_sum / shared_denominator1_sum + shared_nominator2_sum / shared_denominator2_sum
                sign_list = np.array(shared_value_final >= 0)  # Record the judgement.
                new_row_index_list = []
                iter_size = len(sign_list)
                for j in range(iter_size):
                    if sign_list[j]: # True, or the former value is bigger than the latter.
                        new_row_index_list.append(row_index_list[j * 2])
                    else:
                        new_row_index_list.append(row_index_list[j * 2 + 1])
                if len(row_index_list) % 2 != 0: # Odd
                    new_row_index_list.append(row_index_list[-1])
            new_row_index_list = comm.bcast(new_row_index_list, root=2)
            row_index_list = new_row_index_list
        return row_index_list[0], row_idx_dict[row_index_list[0]] # Return feature and split position

    # Implement the First-layer mask and optimize the judgment.
    def SARGMAX_ver4(self, gain_left_up, gain_left_down, gain_right_up, gain_right_down, rank, depth, legal_featureList):
        new_col_index_list = None
        new_row_index_list = None
        row_idx_dict = {}
        row_num = gain_left_up.shape[0]
        nominator_sign_list = denominator_sign_list = None
        for k in range(row_num):
            if depth == 1:  # The first layer.
                if rank == 1:  # The first party who holds labels.
                    if k not in legal_featureList:
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
            col_index_list = [i for i in range(0, len(gain_right_down[0, :]))]
            while len(col_index_list) > 1:
                iter_size = len(col_index_list)
                if iter_size % 2 != 0:  # Odd
                    iter_size -= 1
                list1 = [gain_left_up[k, col_index_list[i]] for i in range(0, iter_size, 2)]
                list2 = [gain_right_down[k, col_index_list[i]] for i in range(0, iter_size, 2)]
                list3 = [gain_right_up[k, col_index_list[i]] for i in range(0, iter_size, 2)]
                list4 = [gain_left_down[k, col_index_list[i]] for i in range(0, iter_size, 2)]

                list5 = [gain_left_up[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list6 = [gain_right_down[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list7 = [gain_right_up[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]
                list8 = [gain_left_down[k, col_index_list[i + 1]] for i in range(0, iter_size, 2)]

                nominator1 = self.SMUL(np.array(list1), np.array(list8), rank) - self.SMUL(np.array(list5),
                                                                                           np.array(list4), rank)
                nominator2 = self.SMUL(np.array(list3), np.array(list6), rank) - self.SMUL(np.array(list7),
                                                                                           np.array(list2), rank)
                denominator1 = self.SMUL(np.array(list4), np.array(list8), rank)
                denominator2 = self.SMUL(np.array(list2), np.array(list6), rank)

                total_nominator = self.SMUL(nominator1, denominator2, rank) + self.SMUL(nominator2, denominator1, rank)
                total_denominator = self.SMUL(denominator1, denominator2, rank)

                total_nominator_list = comm.gather(total_nominator, root=2)
                total_deominator_list = comm.gather(total_denominator, root=1)

                if rank == 2:
                    total_nominator_list = total_nominator_list[1:]
                    nominator_sign_list = np.sum(np.array(total_nominator_list), axis=0)
                    nominator_sign_list[nominator_sign_list >= 0] = 1
                    nominator_sign_list[nominator_sign_list < 0] = -1
                    comm.send(nominator_sign_list, dest=1)
                elif rank == 1:
                    total_denominator_list = total_deominator_list[1:]
                    denominator_sign_list = np.sum(np.array(total_denominator_list), axis=0)
                    denominator_sign_list[denominator_sign_list >= 0] = 1
                    denominator_sign_list[denominator_sign_list < 0] = -1
                    nominator_sign_list = comm.recv(source=2)
                    sign_list = denominator_sign_list * nominator_sign_list # Record the judgement.
                    sign_list = sign_list >= 0 + 0
                    new_col_index_list = []
                    iter_size = len(sign_list)
                    for j in range(iter_size):
                        if sign_list[j]:  # True, or the former value is bigger than the latter.
                            new_col_index_list.append(col_index_list[j * 2])
                        else:
                            new_col_index_list.append(col_index_list[j * 2 + 1])
                    if len(col_index_list) % 2 != 0:  # Odd
                        new_col_index_list.append(col_index_list[-1])
                new_col_index_list = comm.bcast(new_col_index_list, root=1)
                col_index_list = new_col_index_list
            col_idx = col_index_list[0]  # Retrieve out the only col index.
            row_idx_dict[k] = col_idx

        row_index_list = [i for i in row_idx_dict.keys()]
        nominator_sign_list = denominator_sign_list = None
        while len(row_index_list) > 1:
            iter_size = len(row_index_list)
            if len(row_index_list) % 2 != 0:  # Odd
                iter_size -= 1
            list1 = [gain_left_up[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list2 = [gain_right_down[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list3 = [gain_right_up[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list4 = [gain_left_down[row_index_list[i], row_idx_dict[row_index_list[i]]] for i in
                     range(0, iter_size, 2)]
            list5 = [gain_left_up[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list6 = [gain_right_down[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list7 = [gain_right_up[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            list8 = [gain_left_down[row_index_list[i + 1], row_idx_dict[row_index_list[i + 1]]] for i in
                     range(0, iter_size, 2)]
            nominator1 = self.SMUL(np.array(list1), np.array(list8), rank) - self.SMUL(np.array(list5),
                                                                                       np.array(list4),
                                                                                       rank)
            nominator2 = self.SMUL(np.array(list3), np.array(list6), rank) - self.SMUL(np.array(list7),
                                                                                       np.array(list2),
                                                                                       rank)
            denominator1 = self.SMUL(np.array(list4), np.array(list8), rank)
            denominator2 = self.SMUL(np.array(list2), np.array(list6), rank)

            total_nominator = self.SMUL(nominator1, denominator2, rank) + self.SMUL(nominator2, denominator1, rank)
            total_denominator = self.SMUL(denominator1, denominator2, rank)

            total_nominator_list = comm.gather(total_nominator, root=2)
            total_deominator_list = comm.gather(total_denominator, root=1)

            if rank == 2:
                total_nominator_list = total_nominator_list[1:]
                nominator_sign_list = np.sum(np.array(total_nominator_list), axis=0)
                nominator_sign_list[nominator_sign_list >= 0] = 1
                nominator_sign_list[nominator_sign_list < 0] = -1
                comm.send(nominator_sign_list, dest=1)
            elif rank == 1:
                total_denominator_list = total_deominator_list[1:]
                denominator_sign_list = np.sum(np.array(total_denominator_list), axis=0)
                denominator_sign_list[denominator_sign_list >= 0] = 1
                denominator_sign_list[denominator_sign_list < 0] = -1
                nominator_sign_list = comm.recv(source=2)
                sign_list = denominator_sign_list * nominator_sign_list  # Record the judgement.
                sign_list = sign_list >= 0 + 0
                new_row_index_list = []
                iter_size = len(sign_list)
                for j in range(iter_size):
                    if sign_list[j]:  # True, or the former value is bigger than the latter.
                        new_row_index_list.append(row_index_list[j * 2])
                    else:
                        new_row_index_list.append(row_index_list[j * 2 + 1])
                if len(row_index_list) % 2 != 0:  # Odd
                    new_row_index_list.append(row_index_list[-1])
            new_row_index_list = comm.bcast(new_row_index_list, root=1)
            row_index_list = new_row_index_list
        # if rank == 0:
        #     print(row_index_list[0], row_idx_dict[row_index_list[0]])
        return row_index_list[0], row_idx_dict[row_index_list[0]]  # Return feature and split position

    # The initial version of judging the best loss reduction's sign, but will recover the value with random factor.
    def SSIGN(self, data, rank):
        random_num = 0
        result_list = None
        sign = None
        if rank == 1:
            nowTime = datetime.now().strftime("%Y%m%d%H%M%S") # Generate one time data to be the random factor.
            uniqueFactor = int(str(nowTime)[-3:])
            random_num = np.random.rand(1) * uniqueFactor
        random_num = comm.bcast(random_num, root=1)
        result_list = comm.gather(random_num * data, root=0)
        if rank == 0:
            result_sum = np.sum(np.array(result_list[1:]))
            if result_sum > 0:
                sign = '+'
            elif result_sum == 0:
                sign = '='
            else:
                sign = '-'
        sign = comm.bcast(sign, root=0)
        return sign

    def SSIGN_ver2(self, gain_left_up, gain_left_down, gain_right_up, gain_right_down, cgain_up, cgain_down, gamma, rank):
        sign = None
        nominator = self.SMUL(self.SMUL(gain_left_up, gain_right_down, rank) + self.SMUL(gain_left_down, gain_right_up, rank) - self.SMUL(gain_left_down, gain_right_down, rank) * gamma * 2, cgain_down, rank)\
                                - self.SMUL(self.SMUL(gain_left_down, gain_right_down, rank), cgain_up, rank)
        denominator = self.SMUL(self.SMUL(gain_left_down, gain_right_down, rank), cgain_down, rank) * 2
        if rank * rank > 1: # Select rank exclude 0 and 1
            comm.send(nominator, dest=1)
        elif rank == 1:
            for i in range(2, clientNum + 1):
                nominator += comm.recv(source=i)
            if nominator > 0:
                sign = 1
            elif nominator == 0:
                sign = 0
            else:
                sign = -1

        if rank != 0 and rank != 2:
            comm.send(denominator, dest=2)
        elif rank == 2:
            for i in range(1, clientNum + 1):
                if i == 2:
                    pass
                else:
                    denominator += comm.recv(source=i)
            if denominator > 0:
                sign = 1
            elif denominator == 0:
                sign = 0
            else:
                sign = -1

        if rank == 2:
            comm.send(sign, dest=1)
        elif rank == 1:
            sign *= comm.recv(source=2) # Judge the final sign.
            if sign == 1:
                sign = '+'
        sign = comm.bcast(sign, root=1)
        return sign

    def S_GD(self, a, b, rank, lamb):
        temp_a = 0
        shared_step = 0
        coef = 2
        m = coef * lamb
        iter = 0
        if rank != 0:
            temp_a = a.copy() + np.random.uniform(0.1*m, m) * 0.5
            if rank != 1:
                comm.send(temp_a, dest=1)
        if rank == 1:
            for i in range(2, clientNum + 1):
                temp_a += comm.recv(source=i)
            temp_a *= 2 # The a is transmitted as a/2 from each client, we must restore it first.
            shared_step = np.array(1 / temp_a).reshape(-1, 1)
            if temp_a <= clientNum * m:
                max_step = math.log(1e-14, math.e)
                z = temp_a / (clientNum * m)
                iter = max_step / math.log((z * coef - 1) / (z * coef), math.e)
                iter = int(np.ceil(iter))
            else:
                max_step = math.log(1e-14, math.e)
                z = temp_a / (clientNum * m)
                iter1 = max_step / math.log((z * coef - 1) / (z * coef), math.e)
                iter2 = max_step / math.log(1 / z, math.e)
                iter1 = int(np.ceil(iter1))
                iter2 = int(np.ceil(iter2))
                iter = min(iter1, iter2)


        eta = comm.bcast(shared_step, root=1)
        iter = comm.bcast(iter, root=1)
        w = np.array([[0]])
        for j in range(iter):
            wi = w - eta * (2 * self.SMUL(a, w, rank) + b)
            w = wi
        return w
