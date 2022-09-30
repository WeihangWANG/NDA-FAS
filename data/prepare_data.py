import os
import random
from utils import *

# DATA_ROOT = '../CASIA-CeFA-Challenge'
# DATA_ROOT = '../oulu-npu'
# VAL_ROOT = '../oulu-npu/Test_files/Test_files/'
# TRN_ROOT = '../oulu-npu/Train_files/Train_files/'

# TRN_IMGS_DIR = DATA_ROOT + '/Training/'
# TST_IMGS_DIR = DATA_ROOT + '/Testing/'
RESIZE_SIZE = 112
# RESIZE_SIZE = 120

def load_train_list(data_dir):
    print("Loading train data ...")
    ### data_path='/Protocols/Protocols/Protocol_3/Train_6.txt'  # oulu
    # data_dir = os.path.join(DATA_ROOT, data_path)
    list = []
    # f = open(DATA_ROOT + '/4@1_train_3_ft.txt')  # CeFA
    f = open(data_dir)  # oulu
    lines = f.readlines()

    for line in lines:
        # video
        line = line.strip().split(',')
        if int(line[0]) > 0:
            list.append(line)
        # # frame
        # line = line.strip().split(' ')
        # if int(line[1]) > 0:
        #     list.append(line)
    return list

def load_val_list(data_dir):
    print("Loading val data ...")
    ### data_path = '/Protocols/Protocols/Protocol_3/Test_6.txt'  # oulu
    # data_dir = os.path.join(DATA_ROOT, data_path)
    list = []
    # f = open(DATA_ROOT + '/4@1_test_3_rect.txt')  # CeFA
    f = open(data_dir)  # oulu
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(',')
        list.append(line)
    return list

def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if int(tmp[1]) > 0:
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print("# pos : ",len(pos_list))
    print("# neg : ", len(neg_list))
    return pos_list




