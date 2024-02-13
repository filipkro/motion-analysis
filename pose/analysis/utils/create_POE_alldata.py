import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pandas as pd
import os
from scipy.interpolate import interp1d
import scipy
from split_sequence import split_peaks_pad
import matplotlib.pyplot as plt
from datetime import datetime

POE_fields = ['_trunk', '_hip', '_femval', '_KMFP',
              '_fem_med_shank', '_foot']
full_POE = ['Trunk', 'Pelvis', 'Femoral valgus', 'KMFP',
            'Femur-medial-to-shank', 'Foot pronation']

poe_index = 1
POE_lit_name = 'hip'
data_dirs = ('healthy-SLS', 'hipp-SLS', 'marked-SLS', 'musse-SLS',
             'shield-SLS', 'ttb-SLS')
NAME_PATH = '/home/filipkr/Documents/xjob/motion-analysis/names/lit-names-datasets.npy'

names = ['trunk_full', 'hip_full', 'femval_full',
         'kmfp_full', 'fem_med_shank_full',
         'foot_full']
# names = ['trunk_consensus_train', 'hip_consensus_train', 'femval_consensus_train',
#          'kmfp_consensus_train', 'fem_med_shank_consensus_train',
#          'foot_consensus_train']
# names = ['trunk_test', 'hip_test', 'femval_test',
#          'kmfp_test', 'fem_med_shank_test',
#          'foot_test']

POE_lit_name = names[poe_index]

# KPTS = np.array([[6, 0], [12, 0], [14, 0], [16, 0]])
# KPTS = np.array([[5, 0], [6, 0], [11, 1], [12, 1], [20, 0]])
# KPTS = np.array([[5, 0], [5, 1], [6, 0], [6, 1], [11, 0], [11, 1],
#                  [12, 0], [12, 1], [14, 0], [14, 1], [16, 0], [16, 1],
#                  [20, 0], [20, 1]])
# KPTS = np.array([[11, 1], [12, 0], [12, 1], [14, 0]])
# KPTS = np.array([[5, 1], [12, 1]])
KPTS = np.array([[5, 0], [5, 1], [6, 0], [6, 1], [11, 0], [11, 1], [12, 0],
                 [12, 1], [14, 0], [14, 1], [16, 0], [16, 1], [20, 0], [20, 1],
                 [21, 0], [21, 1], [22, 0], [22, 1]])
ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20]]
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[12,0],[20,0]],
                  [[14, 0], [20, 0]]])

KPTS = np.array([[5, 0], [5, 1], [11, 1], [12, 1], [14, 0], [14, 1], [20, 1],
                 [21, 1], [22, 1]])
ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20]]
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[12,0], [20,0]],
                  [[14, 0], [20, 0]]])

KPTS = np.array([[5, 0], [11, 1], [12, 1], [14, 0], [14, 1], [21, 1]])
ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20]]
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[12,0], [20,0]],
                  [[14, 0], [20, 0]]])

KPTS = np.array([[5, 0], [5, 1], [6, 0], [6, 1], [11, 0], [11, 1], [12, 0],
                 [12, 1], [14, 0], [14, 1], [16, 0], [16, 1], [20, 0], [20, 1],
                 [21, 0], [21, 1], [22, 0], [22, 1]])
ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20], [21, 22], [20, 22], [16, 22]]
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[12,0], [20,0]],
                  [[14, 0], [20, 0]], [[20, 0], [22, 0]]])


KPTS = np.array([[5, 1], [6, 1], [14, 1], [16, 1], [20, 1], [21, 1]])                 
ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20], [21, 22], [20, 22], [16, 22]]
DIFFS = np.array([[[12,0], [20,0]], [[14, 0], [20, 0]]])

KPTS = np.array([[6, 1], [14, 1], [20, 1], [21, 1]])                 
ANGLES = [[14, 16], [14, 20], [20, 22], [16, 22]]
DIFFS = np.array([[[14, 0], [20, 0]]])

# KPTS =np.array([[6, 1],[12,  0], [14,  0]])
# ANGLES = [[14,  16]]

# KPTS = np.array([[5, 0], [6, 1], [11, 1], [12, 0], [12, 1]])
# KPTS = np.array([[6, 0], [6, 1], [11, 1], [12,0], [12, 1], [14, 0], [16,1]])
# KPTS = np.array([[6, 0], [6, 1], [11,1], [12, 0],[12,1]])
# KPTS = np.array([[6, 0], [6, 1], [11, 1], [14, 1]])

# KPTS = np.array([[5,0],[6, 0], [6, 1],[11,0], [11,1], [12, 0]])

# KPTS = np.array([[]])
# ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20]]
# ANGLES = [[12, 14],[14,20]]
# ANGLES = [[12, 14], [14,16]]
# ANGLES = [[12, 14]]
# ANGLES = []
# KPTS = 'all'
# ANGLES = [[16, 20]]
# KPTS = np.array([[6, 1], [12, 0], [14, 0]])

KPTS = np.array([[5, 0], [6, 0], [6, 1], [11, 0], [11, 1], [12, 0]])
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [20, 0]]])
ANGLES = []

KPTS = np.array([[5, 1], [12, 1]])
ANGLES = [[16, 20]]
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[14, 0], [20, 0]]])

KPTS = np.array([[5, 0], [5, 1], [11, 1], [12, 1], [14, 0], [14, 1], [20, 1],
                 [21, 1], [22, 1]])
ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20]]
DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[12,0], [20,0]],
                  [[14, 0], [20, 0]]])


KPTS = np.array([[6, 0], [6, 1], [11, 1], [16, 1]])
ANGLES = [[12, 14]]
DIFFS = np.array([[[14,0], [16,0]], [[14, 0], [20, 0]]])
# DIFFS = np.array([[[14, 0], [20, 0]]])
# DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]],[[12,0],[20,0]],
#                   [[14, 0], [20, 0]]])
# DIFFS = np.array([[[14, 0], [20, 0]]])
# DIFFS = np.array([[[12,0],[14,0]]])

# KPTS = np.array([[6, 1], [12, 0], [14, 0]])
# ANGLES = [[14, 16]]
# DIFFS = np.array([[]])

# if len(KPTS) < 1:
#     KPTS =[[]]

if POE_fields[poe_index] == '_trunk':
    KPTS = np.array([[5, 0], [6, 0], [6, 1], [11, 0], [11, 1], [12, 0]])
    DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [20, 0]]])
    ANGLES = []
elif POE_fields[poe_index] == '_hip':
    KPTS = np.array([[6, 0], [6, 1], [11, 1], [16, 1]])
    ANGLES = [[12, 14]]
    DIFFS = np.array([[[14,0], [16,0]], [[14, 0], [20, 0]]])
elif POE_fields[poe_index] == '_femval':
    KPTS = np.array([[6, 1], [12, 0], [14, 0]])
    ANGLES = [[14, 16]]
    DIFFS = np.array([[]])
elif POE_fields[poe_index] == '_KMFP':
    KPTS = np.array([[5, 1], [12, 1]])
    ANGLES = [[16, 20]]
    DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[14, 0], [20, 0]]])
elif POE_fields[poe_index] == '_fem_med_shank':
    KPTS = np.array([[5, 0], [5, 1], [11, 1], [12, 1], [14, 0], [14, 1], [20, 1],
                 [21, 1], [22, 1]])
    ANGLES = [[12, 14], [14, 16], [14, 20], [16, 20]]
    DIFFS = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]], [[12, 0], [20, 0]],
                    [[14, 0], [20, 0]]])
elif POE_fields[poe_index] == '_foot':
    KPTS = np.array([[6, 1], [14, 1], [20, 1], [21, 1]])                 
    ANGLES = [[14, 16], [14, 20], [20, 22], [16, 22]]
    DIFFS = np.array([[[14, 0], [20, 0]]])

lower_peaks = ('hipp12-SLS-L-25.npy', 'marked04-SLS-L-25.npy',
               'musse05-SLS-R-25.npy', 'musse08-SLS-R-25.npy',
               'musse11-SLS-R-25.npy', 'musse10-SLS-R-25.npy')


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
    # poses = np.divide(poses, dist)
    poses = poses - poses[0, 12, :]  # np.expand_dims(poses[0, 0, :], 1)
    return poses


def normalize_coords_motions(poses):
    # print(poses.shape)
    poses = poses - np.mean(poses[:5, 12, :], axis=0)
    poses = poses / np.linalg.norm(np.mean(poses[:5, 5, :], axis=0))
    # dist = np.norm(np.mean(poses[:5, 5, :] - poses[:5, 12, :], axis=0))
    # poses = poses / dist
    # np.expand_dims(poses[0, 0, :], 1)

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


def calc_diffs(poses, kpts):
    max_idx = np.where(poses[:, 0, 0] < -900)[0][0]
    diffs = np.zeros((poses.shape[0], kpts.shape[0]))

    for i in range(kpts.shape[0]):
        diffs[:max_idx, i] = (poses[:max_idx, kpts[i, 0, 0], kpts[i, 0, 1]] -
                              poses[:max_idx, kpts[i, 1, 0], kpts[i, 1, 1]])
    diffs[max_idx:, ...] = np.ones(diffs[max_idx:, ...].shape) \
        * poses[-1, 0, 0]

    return diffs


def get_new_vids(path, dataset, dataset_labels, k, poe,
                 rate=25, ifile=None, debug=False):

    print(full_POE[poe_index])


    labelfile = np.genfromtxt(os.path.join(path, 'new-vids/new-labels.csv'),
                            delimiter=',', dtype=object)

    for file_name in os.listdir(os.path.join(path, 'new-vids')):
        if file_name.endswith('.npy'):
            if args.debug:
                print('new vid')
                print(file_name)

            fps = int(file_name.split('-')[-1].split('.')[0])
            orig_filename = file_name.split('/')[0].replace(f'-{fps}.', '.mp4')
            leg = 'R' if '_R_' in orig_filename else 'L'

            data = np.load(os.path.join(path, file_name))
            data = resample(data, fps / args.rate)
            b, a = scipy.signal.butter(4, 0.2)
            data = scipy.signal.filtfilt(b, a, data, axis=0)
            # plt.plot(data[:,5,1])
            data = normalize_coords(data)

            subj_ind = np.where(labelfile[:,[1,2]] ==
                                [orig_filename, full_POE[poe_index]])[0][0]
            label = int(labelfile[subj_ind, 3])

            angles = calc_angle(data, ANGLES)
            if KPTS.size > 0:
                kpts = data[:, KPTS[:, 0], KPTS[:, 1]].T
            else:
                kpts = np.moveaxis(data[:, [], :], 1, 0)
                kpts = kpts.reshape(kpts.shape[0], -1)

            feats = np.append(kpts, angles, axis=-1)
            if DIFFS.size > 3:
                diffs = calc_diffs(data, DIFFS)
                feats = np.append(feats, diffs, axis=-1)

            dataset.append(feats)
            dataset_labels.append(label)

            if ifile is not None:
                ifile.write('{},{},{},{},{},{}\n'.format(k+1+subj_ind,
                                                         k+1+subj_ind, 0,
                                                         subj_ind, leg,
                                                         'new-vids'))



    return dataset, dataset_labels, ifile


def main(args):
    # labels = pd.read_csv(args.labels, delimiter=',')
    POE_field = POE_fields[poe_index]
    pad = 4 * args.rate

    if args.info_file:
        # lit_names = np.load(NAME_PATH, allow_pickle=True)
        # lit_idx = np.random.choice(lit_names.size)
        # lit_name = lit_names[lit_idx]
        lit_name = POE_lit_name
        coco_data = args.root.split('poses/')[1]

        # lit_name = 'Jean-Paul-Sartre'
        print('The lucky laureate is {}!'.format(lit_name))

        save_path = args.save_path.split('.')[0] + 'data_' + lit_name + '.npz'

        ifile = open(save_path.split('.')[0] + '-info.txt', 'w')
        ifile.write('Dataset {}, created {}, {},,,\n'.format(
            lit_name, datetime.date(datetime.now()),
            str(datetime.time(datetime.now())).split('.')[0]))
        ifile.write(f'COCO dataset used: {coco_data},,,,,\n')
        ifile.write('Action:, {},,,,\n'.format(POE_field))
        ifile.write('FPS:, {},,,,\n'.format(args.rate))
        ifile.write('Keypoints:, {},,,,\n'.format(str(KPTS).replace(',', ' ')))
        ifile.write('Angles:, {},,,,\n \n'.format(
            str(ANGLES).replace(',', ' ')))
        ifile.write('Diffs:, {},,,,\n'.format(str(DIFFS).replace(',', ' ')))
        ifile.write(
            'index,global subject,repetition,subject in cohort,leg,cohort\n')

    dataset_labels = []
    dataset = []
    k = 0
    glob_subject_nbr = 0
    for directory in os.listdir(args.root):
        if directory in data_dirs:
            cohort = directory.split('-')[0]
            dir_path = os.path.join(args.root, directory)
            label_file = os.path.join(dir_path, cohort + '-labels.csv')
            labels = pd.read_csv(label_file, delimiter=',')
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.npy'):
                    if args.debug:
                        print(cohort)
                        print(file_name)
                    subject = int(file_name.split('-')[0])
                    action = file_name.split('-')[1]
                    fps = int(file_name.split('.')[0].split('-')[3])
                    leg = str(file_name.split('-')[2])
                    label_ind = np.where(labels.values[:, 0] == subject)
                    for idx in label_ind[0]:
                        if str(labels.values[idx, 1]).endswith((leg, '0.0',
                                                                '1.0', '2.0')):
                            data = np.load(os.path.join(dir_path, file_name))
                            # idx ???

                            data = resample(data, fps / args.rate)
                            # plt.plot(data[:,5,1])
                            # plt.show()
                            b, a = scipy.signal.butter(4, 0.2)
                            data = scipy.signal.filtfilt(b, a, data, axis=0)
                            # plt.plot(data[:,5,1])
                            data = normalize_coords(data)
                            # plt.plot(data[:,5,1])
                            # plt.show()
                            # if cohort + file_name == 'hipp12-SLS-L-25.npy':
                            #     motions, _ = split_peaks_pad(data, args.rate,
                            #                                  xtra_samp=pad,
                            #                                  joint=5,
                            #                                  debug=args.debug,
                            #                                  prom=0.022)
                            # print(motions.shape)
                            if cohort + file_name in lower_peaks:
                                motions, _ = split_peaks_pad(data, args.rate,
                                                             xtra_samp=pad,
                                                             joint=5,
                                                             prom=0.02,
                                                             debug=args.debug)
                            else:
                                motions, _ = split_peaks_pad(data, args.rate,
                                                             xtra_samp=pad,
                                                             joint=5)

                            if args.debug:
                                print('data shape: {}'.format(data.shape))
                                print('motion shape: {}'.format(motions.shape))

                            # motions = normalize_coords_motions(motions)
                            for i in range(motions.shape[0]):
                                # print(motions[i, :, 0, 0])
                                max_idx = np.where(
                                    motions[i, :, 0, 0] < -900)[0]
                                max_idx = max_idx[0] if len(
                                    max_idx) > 0 else 200
                                motions[i, :max_idx, ...] = normalize_coords_motions(
                                    motions[i, :max_idx, ...])

                            for i in range(5):
                                field = action + POE_field + str(i + 1)
                                label = labels.filter(like=field).values[idx]
                                if label.size > 0 and not np.isnan(label):
                                    if False:
                                        feats = motions[i, ...]
                                        feats = feats.reshape(
                                            feats.shape[0], -1)
                                    else:
                                        # print(f'cohort: {cohort}, sub: {subject}, leg: {leg}, rep: {i}')
                                        # print(motions.shape)
                                        # print(i)
                                        # print(label)
                                        # print(field)
                                        # print(subject)
                                        # print(action)
                                        # print(leg)
                                        # print(cohort)
                                        # print(motions[i, ...].shape)
                                        angles = calc_angle(
                                            motions[i, ...], ANGLES)
                                        if KPTS.size > 0:
                                            kpts = motions[i, :,
                                                           KPTS[:, 0], KPTS[:, 1]].T
                                        else:
                                            kpts = np.moveaxis(
                                                motions[i, :, [], :], 1, 0)
                                            kpts = kpts.reshape(
                                                kpts.shape[0], -1)
                                        feats = np.append(
                                            kpts, angles, axis=-1)
                                        if DIFFS.size > 3:
                                            diffs = calc_diffs(
                                                motions[i, ...], DIFFS)
                                            feats = np.append(feats, diffs,
                                                              axis=-1)

                                    dataset.append(feats)
                                    dataset_labels.append(label)

                                    if args.info_file:
                                        ifile.write('{},{},{},{},{},{}\n'.format(
                                            k, glob_subject_nbr, i, subject, leg, cohort))
                                    k += 1

                    if str(labels.values[idx, 1]).endswith((leg, '0.0',
                                                            '1.0', '2.0')):
                        glob_subject_nbr += 1

    dataset_labels = np.array(dataset_labels)
    dataset = np.array(dataset)
    print(dataset.shape)

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
        # lit_names = np.delete(lit_names, lit_idx)
        # print('Names left: {}'.format(lit_names.size))
        # np.save(NAME_PATH, lit_names)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('labels')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--debug', type=str2bool, nargs='?', default=False)
    parser.add_argument('--rate', type=int, default=25)
    parser.add_argument('--info_file', type=str2bool, nargs='?', default=True)
    parser.add_argument('--gen_idx', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    if args.save_path == '':
        args.save_path = os.path.join(args.root, 'dataset.npz')

    main(args)
