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

    max_idx = np.where(poses[:, 0, 0] < -900)[0][0]

    for i in range(len(kpts)):
        angles[:max_idx, i] = np.arctan2(poses[:max_idx, kpts[i][0], 1] -
                                         poses[:max_idx, kpts[i][1], 1],
                                         poses[:max_idx, kpts[i][0], 0] -
                                         poses[:max_idx, kpts[i][1], 0])
    angles[max_idx:, ...] = np.ones(angles[max_idx:, ...].shape) \
        * poses[-1, 0, 0]

    return np.array(angles)


def main(args):
    labels = pd.read_csv(args.labels, delimiter=',')
    POE_field = POE_fields[2]
    pad = 4 * args.rate

    if args.info_file:
        lit_names = np.load(NAME_PATH, allow_pickle=True)
        lit_idx = np.random.choice(lit_names.size)
        lit_name = lit_names[lit_idx]
        lit_names = np.delete(lit_names, lit_idx)
        print('The lucky laureate is {}!'.format(lit_name))
        print('Names left: {}'.format(lit_names.size))
        # np.save(NAME_PATH, lit_names)
        save_path = args.save_path.split('.')[0] + 'data_' + lit_name + '.npz'

        ifile = open(save_path.split('.')[0] + '-info.txt', 'w')
        ifile.write('Dataset {}, created {}, {}\n'.format(
            lit_name, datetime.date(datetime.now()),
            str(datetime.time(datetime.now())).split('.')[0]))
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
    for file_name in os.listdir(args.data_folder):
        if file_name.endswith('.npy'):
            if args.debug:
                print(file_name)
            data = np.load(os.path.join(args.data_folder, file_name))
            subject = int(file_name.split('-')[0])
            action = file_name.split('-')[1]
            fps = int(file_name.split('.')[0].split('-')[3])
            data = resample(data, fps / args.rate)
            data = normalize_coords(data)
            motions, _ = split_peaks_pad(data, args.rate,
                                         xtra_samp=pad, joint=5)
            for i in range(5):

                field = action + POE_field + str(i + 1)
                label = labels.filter(like=field).values[subject - 1][0]

                if not np.isnan(label):
                    angles = calc_angle(motions[i, ...], ANGLES)
                    # kpts = np.moveaxis(motions[i, :, KPTS, :], 1, 0)
                    # kpts = kpts.reshape(kpts.shape[0], -1)
                    if KPTS.size > 0:
                        kpts = motions[i, :, KPTS[:, 0], KPTS[:, 1]].T
                    else:
                        kpts = np.moveaxis(motions[i, :, [], :], 1, 0)
                        kpts = kpts.reshape(kpts.shape[0], -1)
                    # print(motions[i,:,KPTS].shape)
                    # print(kpts - motions[i,:,KPTS])
                    feats = np.append(kpts, angles, axis=-1)
                    dataset.append(feats)
                    dataset_labels.append(label)

                    if subject in TV_subj and args.gen_idx:
                        if t % 2 == 0:
                            vi = np.append(vi, k)
                        else:
                            ti = np.append(ti, k)
                        t += 1
                    elif args.gen_idx:
                        train_idx = np.append(train_idx, k)
                    if args.info_file:
                        ifile.write('{},{},{}\n'.format(k, subject, i))
                    k += 1

    dataset_labels = np.array(dataset_labels)
    dataset = np.array(dataset)

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

    if args.gen_idx:
        np.savez('/home/filipkr/Documents/xjob/data/datasets/indices.npz',
                 train_idx=train_idx, test_idx=ti, val_idx=vi)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('labels')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--debug', type=str2bool, nargs='?', default=False)
    parser.add_argument('--rate', type=int, default=25)
    parser.add_argument('--info_file', type=str2bool, nargs='?', default=True)
    parser.add_argument('--gen_idx', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    if args.save_path == '':
        args.save_path = os.path.join(args.data_folder, 'dataset.npz')

    main(args)
