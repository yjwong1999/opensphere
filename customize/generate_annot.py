import os
import scipy.io

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, required=True, help='Main directory for the dataset') 

opt = parser.parse_args()

# train
data_dir = opt.directory
train_data_dir = os.path.join(data_dir, 'training_set')
train_data_annot = os.path.join(data_dir, f'{data_dir}_train_ann.txt')

if os.path.isfile(train_data_annot):
    os.remove(train_data_annot)
    
with open(train_data_annot, "a") as f:
    for name_dir in os.listdir(train_data_dir):
        name = name_dir
        name_dir = os.path.join(train_data_dir, name_dir)
        for path in os.listdir(name_dir):
            path = os.path.join(os.getcwd(), name_dir, path)
            txt_line = path + ' ' + f'id_{name}'
            f.write(txt_line + "\n")
        
# test
test_data_dir = os.path.join(data_dir, 'Face_Verification_Test_Set/verification_images')
test_data_annot = os.path.join(data_dir, f'{data_dir}_test_ann.txt')

if os.path.isfile(test_data_annot):
    os.remove(test_data_annot)
    
ori_data_annot_pos = os.path.join(data_dir, 'Face_Verification_Test_Set/positive_pairs_names.mat')
ori_data_annot_neg = os.path.join(data_dir, 'Face_Verification_Test_Set/negative_pairs_names.mat')

with open(test_data_annot, "a") as f: 
    # positive pairs
    mat = scipy.io.loadmat(ori_data_annot_pos)
    pos_pairs = mat['positive_pairs_names']
    for pair in pos_pairs:
        path_1 = os.path.join(test_data_dir, pair[0][0])
        path_2 = os.path.join(test_data_dir, pair[1][0])
        txt_line = '1' + ' ' + path_1 + ' ' + path_2
        f.write(txt_line + "\n")

    # negative pairs
    mat = scipy.io.loadmat(ori_data_annot_neg)
    neg_pairs = mat['negative_pairs_names']
    for pair in neg_pairs:
        path_1 = os.path.join(test_data_dir, pair[0][0])
        path_2 = os.path.join(test_data_dir, pair[1][0])
        txt_line = '0' + ' ' + path_1 + ' ' + path_2
        f.write(txt_line + "\n")


'''
import os
import scipy.io

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, required=True, help='Main directory for the dataset') 

opt = parser.parse_args()

# train
data_dir = opt.directory
train_data_dir = os.path.join(data_dir, 'training_set')
train_data_annot = os.path.join(data_dir, f'{data_dir}_train_ann.txt')

if os.path.isfile(train_data_annot):
    os.remove(train_data_annot)
    
with open(train_data_annot, "a") as f:
    for name_dir in os.listdir(train_data_dir):
        name = name_dir
        name_dir = os.path.join(train_data_dir, name_dir)
        for path in os.listdir(name_dir):
            path = os.path.join(os.getcwd(), name_dir, path)
            txt_line = path + ' ' + f'id_{name}'
            f.write(txt_line + "\n")
'''
