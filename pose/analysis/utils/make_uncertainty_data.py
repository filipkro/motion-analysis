import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pandas as pd
import os
import scipy
from split_sequence import split_peaks_pad
import matplotlib.pyplot as plt
from datetime import datetime

POE_fields = ['_trunk', '_hip', '_femval', '_KMFP',
              '_fem_med_shank', '_foot']
possible_indx = [0,1,2,3]
poe_index = 2
data_dirs = ('healthy-SLS', 'hipp-SLS', 'marked-SLS', 'musse-SLS',
             'shield-SLS', 'ttb-SLS')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx, 1]
    in_cohort_nbr = data[idx, 3]
    indices = np.where(data[:, 1] == subj)[0]
    idx_same_leg = np.where(data[indices, 3] == in_cohort_nbr)
    # print(indices[idx_same_leg])
    return indices[idx_same_leg]


def get_subject_index(info_file, cohort, subject, leg):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :], dtype=str)
    # print(data)
    # print(data[:,-1])
    data = data[np.where(data[:,-1] == cohort)[0], :]
    # print(data)
    data = data[np.where(data[:,3] == str(subject))[0], :]
    # print(data)
    data = data[np.where(data[:,-2] == leg)[0], :]
    # print(data)
    return int(data[0,1])

def get_nbr_subjects(info_file):
    meta_data = pd.read_csv(info_file, delimiter=',')
    return int(meta_data.values[-1, 1]) + 1


def main(args):
    # labels = pd.read_csv(args.labels, delimiter=',')
    assert args.poe_index in possible_indx

    POE_field = POE_fields[args.poe_index]

    if POE_field == '_femval':
        info_file = '/home/filipkr/Documents/xjob/data/datasets/data_Paul-Heyse-info.txt'
    elif POE_field == '_trunk':
        info_file = '/home/filipkr/Documents/xjob/data/datasets/data_Miguel-Angel-Asturias-info.txt'
    elif POE_field == '_hip':
        info_file = '/home/filipkr/Documents/xjob/data/datasets/data_Elfriede-Jelinek-info.txt'
    elif POE_field == '_KMFP':
        info_file = '/home/filipkr/Documents/xjob/data/datasets/data_Mikhail-Sholokhov-info.txt'

    action = 'SLS'
    uncertainty_dict = np.zeros(get_nbr_subjects(info_file))
    for dir in os.listdir(args.root):
        if dir in data_dirs:
            cohort = dir.split('-')[0]
            dir_path = os.path.join(args.root, dir)
            label_file = os.path.join(dir_path, cohort + '-labels.csv')
            labels = pd.read_csv(label_file, delimiter=',')

            field = action + POE_field + '_uncertainty'
            subjects = labels.values[:, 0]
            legs = labels.values[:, 1]
            uncertainty = labels.filter(like=field).values

            # print(cohort)
            # print(subjects)
            # print(legs)
            # print(uncertainty)

            for subject, leg, unc in zip(subjects, legs, uncertainty):
                if unc in [1,2,3]:
                    idx = get_subject_index(info_file, cohort, subject, leg)
                    uncertainty_dict[idx] = unc

    # print(uncertainty_dict)
    file_name = 'uncertainties-' + POE_field.split('_')[-1] + '.npy'
    np.save(os.path.join(args.save_path, file_name), uncertainty_dict)
    print('--DONE--')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('poe_index', type=int)
    parser.add_argument('--save_path', default='')
    args = parser.parse_args()

    main(args)
