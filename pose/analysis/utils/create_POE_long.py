import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pandas as pd
import os
from scipy.interpolate import interp1d
from split_sequence import split_peaks_pad
import matplotlib.pyplot as plt
from datetime import datetime

POE_fields = ['_trunk', '_hip', '_femval', '_KMFP',
              '_fem_med_shank', '_foot']
NAME_PATH = '/home/filipkr/Documents/xjob/motion-analysis/names/lit-names-datasets.npy'

# KPTS = np.array([[6, 0], [12, 0], [14, 0], [16, 0]])
KPTS = np.array([[]])
KPTS = 'all'
ANGLES = [[12, 14], [14, 16]]  # , [14, 16]]

# if len(KPTS) < 1:
#     KPTS =[[]]
TV_subj = [3, 7, 12, 16, 24, 25, 38, 52]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def normalize_coords(poses):
    dist = abs(poses[10, 0, 1] - poses[10, 16, 1])
    poses = poses / dist
    poses = poses - poses[0, 12, :]  # np.expand_dims(poses[0, 0, :], 1)
    return poses


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def calc_angle(poses, kpts):
    angles = np.zeros((poses.shape[0], len(kpts)))

    max_idx = np.where(poses[:, 0, 0] < -900)[0]
    max_idx = max_idx[0] if len(max_idx) > 0 else poses.shape[0]

    for i in range(len(kpts)):
        angles[:max_idx, i] = np.arctan2(poses[:max_idx, kpts[i][0], 1] -
                                         poses[:max_idx, kpts[i][1], 1],
                                         poses[:max_idx, kpts[i][0], 0] -
                                         poses[:max_idx, kpts[i][1], 0])
    angles[max_idx:, ...] = np.ones(angles[max_idx:, ...].shape) \
        * poses[-1, 0, 0]

    return np.array(angles)


def main(args):
    POE_field = 'SLS' + POE_fields[2] + '_mean'
    pad_len = 1100

    if args.info_file:
        lit_names = np.load(NAME_PATH, allow_pickle=True)
        lit_idx = np.random.choice(lit_names.size)
        lit_name = lit_names[lit_idx]
        lit_names = np.delete(lit_names, lit_idx)
        # lit_name = 'Czeslaw-Milosz'
        print('The lucky laureate is {}!'.format(lit_name))
        print('Names left: {}'.format(lit_names.size))
        np.save(NAME_PATH, lit_names)
        save_path = args.save_path.split('.')[0] + 'data_' + lit_name + '.npz'

        ifile = open(save_path.split('.')[0] + '-info.txt', 'w')
        ifile.write('Dataset {}, created {}, {}\n'.format(
            lit_name, datetime.date(datetime.now()),
            str(datetime.time(datetime.now())).split('.')[0]))
        ifile.write('Sequence with all reps, label is mean score,\n')
        ifile.write('Sequences are padded to length, {},\n'.format(pad_len))
        ifile.write('Action:, {},\n'.format(POE_field))
        ifile.write('FPS:, {},\n'.format(args.rate))
        ifile.write('Keypoints:, {},\n'.format(str(KPTS).replace(',', ' ')))
        ifile.write('Angles:, {},\n \n'.format(str(ANGLES).replace(',', ' ')))
        ifile.write('index,subject,repetition\n')

    dataset_labels = []
    dataset = []
    k = 0
    t = 0
    ti = np.array([])
    vi = np.array([])
    train_idx = np.array([])

    for dir in os.listdir(args.root_folder):
        if dir != 'orig-poses':
            dir_path = os.path.join(args.root_folder, dir)
            label_file = os.path.join(
                dir_path, dir.split('-')[0] + '-labels.csv')
            labels = pd.read_csv(label_file, delimiter=',')
            # if str(labels.values[0,1]).endswith(('L','R')):
            # print(labels.values[:,1])

            for file_name in os.listdir(dir_path):
                if file_name.endswith('.npy'):
                    subject = int(file_name.split('-')[0])
                    fps = int(file_name.split('.')[0].split('-')[3])
                    leg = str(file_name.split('-')[2])
                    label_ind = np.where(labels.values[:, 0] == subject)
                    for idx in label_ind[0]:
                        # print(str(labels.values[idx,1]))
                        if str(labels.values[idx, 1]).endswith((leg, '0.0',
                                                                '1.0', '2.0')):
                            # print(dir)
                            # print(file_name)
                            # print(labels.values[idx, 1])
                            # print(labels.filter(like=POE_field).values[idx])
                            # print('')
                            label = labels.filter(like=POE_field).values[idx]

                            data = np.load(os.path.join(dir_path, file_name))
                            data = resample(data, fps / args.rate)
                            data = normalize_coords(data)
                            # print('line 127 (data): ', data.shape)
                            if not np.isnan(label):
                                pad = pad_len - data.shape[0]
                                angles = calc_angle(data, ANGLES)
                                angles = np.pad(angles, ((0, pad), (0, 0)),
                                                'constant',
                                                constant_values=-1000)
                                # print('line 130: (angles)', angles.shape)
                                # print(KPTS.size)
                                feats = []
                                if KPTS == 'all':
                                    feats = data.reshape(data.shape[0], -1)
                                    feats = np.pad(feats,((0,pad), (0,0)), 'constant', constant_values=-1000)
                                elif KPTS.size > 0:
                                    kpts = data[:, KPTS[:, 0], KPTS[:, 1]]
                                    kpts = np.pad(kpts, ((0, pad), (0, 0)),
                                                  'constant',
                                                  constant_values=-1000)
                                    feats = np.append(kpts, angles, axis=-1)

                                feats = np.append(feats, angles, axis=-1)

                                # print('line 137 (kpts)', kpts.shape)
                                # kpts = np.pad(kpts, ((0, pad), (0, 0)),
                                #               'constant',
                                #               constant_values=-1000)
                                # # print('line 141 (kpts)', kpts.shape)
                                # angles = np.pad(angles, ((0, pad), (0, 0)),
                                #                 'constant',
                                #                 constant_values=-1000)
                                # print('line 144: (angles)', angles.shape)

                                dataset.append(feats)
                                # print('feat shape:', feats.shape)
                                dataset_labels.append(label)

    dataset_labels = np.array(dataset_labels)
    dataset = np.array(dataset)
    print('label shape', dataset_labels.shape)
    print('dataset shape', dataset.shape)

    if args.debug:
        print(dataset.shape)
        print(dataset_labels.shape)
        print(dataset)

        # plt.plot(dataset[..., 2].T)
        # plt.ylim((-2, 0))
        # plt.figure()
        plt.plot(dataset[..., 0].T)
        plt.ylim((-2, 0))
        plt.show()

    if args.info_file:
        np.savez(save_path, mts=dataset, labels=dataset_labels)
        ifile.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root_folder')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--debug', type=str2bool, nargs='?', default=False)
    parser.add_argument('--rate', type=int, default=25)
    parser.add_argument('--info_file', type=str2bool, nargs='?', default=True)
    parser.add_argument('--gen_idx', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    if args.save_path == '':
        args.save_path = os.path.join(args.root_folder, 'dataset.npz')

    main(args)
