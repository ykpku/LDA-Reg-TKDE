# -*- coding:utf-8 -*-
import re
import pandas as pd
import os
import cPickle as CPickle
import numpy as np
import time
from scipy import sparse

class CsvUtility(object):
    @staticmethod
    def write2pickle(file_path, object, type):
        with open(file_path, type) as f:
            CPickle.dump(object, f)
            f.close()

    @staticmethod
    def read_pickle(file_path, type):
        with open(file_path, type) as f:
            obj = CPickle.load(f)
            f.close()
        return obj

    @staticmethod
    def write_dict2csv(raw_dict, csv_path, file_name):
        pd.DataFrame.from_dict(raw_dict, orient='index').to_csv(os.path.join(csv_path, file_name), header=False)

    @staticmethod
    def write_key_value_times(raw_dict, csv_path, file_name):
        with open(csv_path + '/' + file_name, 'w') as f:
            write_str = ""
            for (key, value) in raw_dict.items():
                for i in range(value):
                    write_str += key + ","
            f.write(write_str)
        return len(raw_dict)

    @staticmethod
    def write_array2csv(raw_array, csv_path, file_name, type='float'):
        np_raw = np.array(raw_array)
        if np_raw.ndim == 1:
            np_raw = np_raw.reshape((len(np_raw), -1))
        pd.DataFrame(np_raw).to_csv(os.path.join(csv_path, file_name), index=None, header=None)

    @staticmethod
    def read_array_from_csv(csv_path, file_name, type='float'):
        re_array = np.array(pd.read_csv(os.path.join(csv_path, file_name), header=None, index_col=None))
        if re_array.shape[0] == 1:
            return re_array.flatten()
        return re_array

    @staticmethod
    def write_list2csv(list_data, csv_path, file_name):
        with open(os.path.join(csv_path, file_name), 'w') as f:
            for item in list_data:
                f.write(str(item))
                f.write('\n')

    @staticmethod
    def convert_dense_2sparse(csv_path, dense_file, sparse_file):
        data = []
        with open(os.path.join(csv_path, dense_file), 'r') as csvfile:
            for row in csvfile.readlines():  # 将csv 文件中的数据保存到birth_data中
                data.append([float(x) for x in row.split(",")])
        sp_data_array = sparse.csr_matrix(np.array(data))  # 将list数组转化成array数组便于查看数据结构
        sparse.save_npz(os.path.join(csv_path, sparse_file), sp_data_array)  # 保存
        print("--convert success!--")
        print("sparse file name: ", sparse_file)

    @staticmethod
    def read_sparse_array_from_csv(csv_path, file_name):
        sp_data = sparse.load_npz(os.path.join(csv_path, file_name))
        data = sp_data.toarray()
        # print(data.shape)  # 利用.shape查看结构。

        # if data_array.shape[0] == 1:
        #     return data_array.flatten()
        return data

if __name__ == '__main__':

    tests = ["dd", "df3ef", "dfeww"]
    print(tests)
    # t1 = time.time()
    # s = CsvUtility.read_array_from_csv("/home1/yk/new_mimic_formal_data/", "formal_train_x_seq.csv")
    # t2 = time.time()
    # print(s.shape, t2-t1)
    CsvUtility.convert_dense_2sparse("/home1/yk/new_mimic_formal_data/", "formal_valid_x_seq.csv", "sparse_formal_valid_x_seq")
    CsvUtility.convert_dense_2sparse("/home1/yk/new_mimic_formal_data/", "formal_test_x_seq.csv", "sparse_formal_test_x_seq")
    # t1 = time.time()
    # f = CsvUtility.fast_read_array_from_csv("/home1/yk/new_mimic_formal_data/", "formal_train_x_seq.csv")
    # t2 = time.time()
    # print(len(f), t2 - t1)
    # print(f[0][:10])
