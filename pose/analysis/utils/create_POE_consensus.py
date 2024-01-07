import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pandas as pd
import os
from scipy.interpolate import interp1d
import scipy
from split_sequence import split_peaks_pad
import matplotlib.pyplot as plt
from datetime import datetime
import csv

POE_fields = ['_trunk', '_hip', '_femval', '_KMFP',
              '_fem_med_shank', '_foot']
full_POE = ['Trunk', 'Pelvis', 'Femoral valgus', 'KMFP',
            'Femur-medial-to-shank', 'Foot pronation']

poe_index = 5
start_index = [520, 519, 530, 530, 520, 518]
start_subj = [104, 104, 104, 104, 104, 104]
names = ['trunk_test', 'hip_test', 'femval_test',
         'kmfp_test', 'fem_med_shank_test',
         'foot_test']
# names = ['trunk_consensus_train', 'hip_consensus_train', 'femval_consensus_train',
#          'kmfp_consensus_train', 'fem_med_shank_consensus_train',
#          'foot_consensus_train']
# POE_lit_name = 'hip_consensus_test'
POE_lit_name = names[poe_index]
data_dirs = ('healthy-SLS', 'hipp-SLS', 'marked-SLS', 'musse-SLS',
             'shield-SLS', 'ttb-SLS')
NAME_PATH = '/home/filipkr/Documents/xjob/motion-analysis/names/lit-names-datasets.npy'

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

    try:
        max_idx = np.where(poses[:, 0, 0] < -900)[0][0]
    except IndexError:
        max_idx = len(poses[:, 0, 0])

    for i in range(len(kpts)):
        angles[:max_idx, i] = np.arctan2(poses[:max_idx, kpts[i][0], 1] -
                                         poses[:max_idx, kpts[i][1], 1],
                                         poses[:max_idx, kpts[i][0], 0] -
                                         poses[:max_idx, kpts[i][1], 0])
    angles[max_idx:, ...] = np.ones(angles[max_idx:, ...].shape) \
        * poses[-1, 0, 0]

    return np.array(angles)


def calc_diffs(poses, kpts):
    try:
        max_idx = np.where(poses[:, 0, 0] < -900)[0][0]
    except:
        max_idx = len(poses[:, 0, 0])

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

    q = 0
    ind_map = {}

    with open(os.path.join(path, 'new-labels.csv')) as f:
        data = list(csv.reader(f, delimiter=','))

    # print(data)
    labelfile = np.array(data)
    # print(data.shape)


    # labelfile = np.genfromtxt(os.path.join(path, 'new-labels.csv'),
    #                         delimiter=',', dtype=object)

    for file_name in os.listdir(path):
        if file_name.endswith('.npy'):
            if args.debug:
                print('new vid')
                print(file_name)

            fps = int(file_name.split('-')[-1].split('.')[0])
            orig_filename = file_name.split('/')[0].replace(f'-{fps}.npy', '.mp4')
            leg = 'R' if '_R_' in orig_filename else 'L'
            # print(file_name)
            # print(fps)
            # print(orig_filename)
            # print(leg)
            data = np.load(os.path.join(path, file_name))
            data = resample(data, fps / args.rate)
            b, a = scipy.signal.butter(4, 0.2)
            data = scipy.signal.filtfilt(b, a, data, axis=0)
            # plt.plot(data[:,5,1])
            data = normalize_coords(data)
            meta = orig_filename.split('.')[0].split('_')
            rep = meta[-1]
            ind = meta.index(leg)
            key = '_'.join(meta[:ind])
            print(key)
            try:
                subj_ind = ind_map[key]
            except KeyError:
                ind_map[key] = q
                subj_ind = q
                q += 1
            # print(labelfile[:,1:])
            # print(labelfile.shape)
            # print([orig_filename, full_POE[poe_index]])
            subj_ind = np.where(labelfile[:,[1,2]] ==
                                [orig_filename, full_POE[poe_index]])[0][0]
            print(subj_ind)
            label = int(labelfile[subj_ind, 3])

            angles = calc_angle(data, ANGLES)

            # print(data.shape)
            # print(KPTS)
            # print(ANGLES)
            if KPTS.size > 0:
                kpts = data[:, KPTS[:, 0], KPTS[:, 1]]
            else:
                kpts = np.moveaxis(data[:, [], :], 1, 0)
                kpts = kpts.reshape(kpts.shape[0], -1)

            # print(angles.shape)
            # print(kpts.shape)
            feats = np.append(kpts, angles, axis=-1)
            if DIFFS.size > 3:
                diffs = calc_diffs(data, DIFFS)
                feats = np.append(feats, diffs, axis=-1)

            pad = 200 - feats.shape[0]
            # print(pad)
            # print(feats.shape)
            feats = np.pad(feats, ((0, pad), (0, 0)), 'constant',
                           constant_values=-1000)
            # print(feats.shape)
            # print(feats)
            # print(feats[:,0])
            # assert False

            dataset.append(feats)
            dataset_labels.append(label)

            if ifile is not None:
                ifile.write('{},{},{},{},{},{}\n'.format(k, subj_ind, rep,
                                                         subj_ind, leg,
                                                         'new-vids'))

                k += 1



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

        save_path = os.path.join(args.save_path.split('.')[0], 'data_' + lit_name + '.npz')
        # print(save_path)
        # print(args.save_path)
        ifile = open(save_path.split('.')[0] + '-info.txt', 'w')
        ifile.write('Dataset {}, created {}, {},,,,\n'.format(
            lit_name, datetime.date(datetime.now()),
            str(datetime.time(datetime.now())).split('.')[0]))
        ifile.write(f'COCO dataset used: {coco_data},,,,,,\n')
        ifile.write('Action:, {},,,,,\n'.format(POE_field))
        ifile.write('FPS:, {},,,,,\n'.format(args.rate))
        ifile.write('Keypoints:, {},,,,,\n'.format(str(KPTS).replace(',', ' ')))
        ifile.write('Angles:, {},,,,,\n \n'.format(
            str(ANGLES).replace(',', ' ')))
        ifile.write('Diffs:, {},,,,,\n'.format(str(DIFFS).replace(',', ' ')))
        ifile.write(
            'index,global subject,repetition,subject in cohort,leg,cohort,full\n')

    dataset_labels = []
    dataset = []
    k = start_index[poe_index]
    glob_subject_nbr = 0

    q = start_subj[poe_index]
    ind_map = {}
    path = args.root

    with open(os.path.join(path, 'new-labels.csv')) as f:
        data = list(csv.reader(f, delimiter=','))

    # print(data)
    labelfile = np.array(data)
    # print(ifile)

    for file_name in sorted(os.listdir(path)):
        # print(file_name)
        if file_name.endswith('.npy') and not ('TTB' in file_name or 'SHIELD' in file_name):
            if args.debug:
                print('new vid')
                print(file_name)

            fps = int(file_name.split('-')[-1].split('.')[0])
            orig_filename = file_name.split('/')[0].replace(f'-{fps}.npy', '.mp4')
            leg = 'R' if '_R_' in orig_filename else 'L'
            data = np.load(os.path.join(path, file_name))
            data = resample(data, fps / args.rate)
            b, a = scipy.signal.butter(4, 0.2)
            data = scipy.signal.filtfilt(b, a, data, axis=0)
            data = normalize_coords(data)
            meta = orig_filename.split('.')[0].split('_')
            rep = meta[-1]
            ind = meta.index(leg)
            key = '_'.join(meta[:ind])
            # print(key)
            try:
                subj_ind = ind_map[key]
            except KeyError:
                ind_map[key] = q
                subj_ind = q
                q += 1
            # label_ind = subj_ind = np.where(labelfile[:,[1,2]] ==
            #                     [orig_filename, full_POE[poe_index]])
            # print(label_ind)
            # # print(labelfile[:,[1,2]])
            # print([orig_filename, full_POE[poe_index]])
            label = None
            for row in labelfile:
                if row[1] == orig_filename and row[2] == full_POE[poe_index]:
                    label = int(row[3])
                    break
            # subj_ind = np.where(labelfile[:,[1,2]] ==
            #                     [orig_filename, full_POE[poe_index]])[0][0]
            # print(subj_ind)

            # print(label)
            if label is None:
                print(orig_filename, full_POE[poe_index])
            # print(subj_ind)

            angles = calc_angle(data, ANGLES)

            if KPTS.size > 0:
                kpts = data[:, KPTS[:, 0], KPTS[:, 1]]
            else:
                kpts = np.moveaxis(data[:, [], :], 1, 0)
                kpts = kpts.reshape(kpts.shape[0], -1)

            feats = np.append(kpts, angles, axis=-1)
            if DIFFS.size > 3:
                diffs = calc_diffs(data, DIFFS)
                feats = np.append(feats, diffs, axis=-1)

            pad = 200 - feats.shape[0]
            feats = np.pad(feats, ((0, pad), (0, 0)), 'constant',
                           constant_values=-1000)

            dataset.append(feats)
            dataset_labels.append(label)

            if ifile is not None:
                ifile.write('{},{},{},{},{},{},{}\n'.format(k, subj_ind, rep,
                                                         subj_ind, leg,
                                                         key, orig_filename))

                k += 1


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