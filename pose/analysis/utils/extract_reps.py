import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import os
from scipy.interpolate import interp1d
import scipy
from split_sequence import split_peaks_pad

femval_kpts = np.array([[6, 1], [12, 0], [14, 0]])
femval_angles = [[14, 16]]
femval_diffs = np.array([[]])

trunk_kpts = np.array([[5, 0], [6, 0], [6, 1], [11, 0], [11, 1], [12, 0]])
trunk_angles = []
trunk_diffs = np.array([[[12, 0], [14, 0]]])

hip_kpts = np.array([[6, 0], [6, 1], [11, 1], [16, 1]])
hip_angles = [[12, 14]]
hip_diffs = np.array([[[14, 0], [16, 0]], [[14, 0], [20, 0]]])

kmfp_kpts = np.array([[5, 1], [12, 1]])
kmfp_angles = [[16, 20]]
kmfp_diffs = np.array([[[12, 0], [14, 0]], [[14, 0], [16, 0]],
                       [[14, 0], [20, 0]]])

all_kpts = [femval_kpts, trunk_kpts, hip_kpts, kmfp_kpts]
all_angles = [femval_angles, trunk_angles, hip_angles, kmfp_angles]
all_diffs = [femval_diffs, trunk_diffs, hip_diffs, kmfp_diffs]

femval_data = []
trunk_data = []
hip_data = []
kmfp_data = []
all_data = [femval_data, trunk_data, hip_data, kmfp_data]

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


def main(args, data, fps):

    pad = 4 * args.rate

    data = resample(data, fps / args.rate)
    # plt.plot(data[:,5,1])
    # plt.show()
    b, a = scipy.signal.butter(4, 0.2)
    data = scipy.signal.filtfilt(b, a, data, axis=0)

    data = normalize_coords(data)
    motions, _ = split_peaks_pad(data, args.rate, xtra_samp=pad, joint=5)

    # if cohort + file_name in lower_peaks:
    #     motions, _ = split_peaks_pad(data, args.rate,
    #                                  xtra_samp=pad,
    #                                  joint=5,
    #                                  prom=0.02,
    #                                  debug=args.debug)

    if args.debug:
        print('data shape: {}'.format(data.shape))
        print('motion shape: {}'.format(motions.shape))

    datasets = {'femval': [], 'trunk': [], 'hip': [], 'kmfp': []}
    poes = ['femval', 'trunk', 'hip', 'kmfp']
    for KPTS, ANGLES, DIFFS, poe in zip(all_kpts, all_angles, all_diffs, poes):
        print(DIFFS)
        dataset = []
        for i in range(motions.shape[0]):
            angles = calc_angle(motions[i, ...], ANGLES)
            if KPTS.size > 0:
                kpts = motions[i, :, KPTS[:, 0], KPTS[:, 1]].T
            feats = np.append(kpts, angles, axis=-1)
            if DIFFS.size > 0:
                diffs = calc_diffs(motions[i, ...], DIFFS)
                feats = np.append(feats, diffs, axis=-1)

            dataset.append(feats)

            # print(feats.shape)
        # datasets[poe].append(np.array(dataset))
        datasets[poe].append(dataset)

    for poe in poes:
        datasets[poe] = np.array(datasets[poe])[0, ...]

    datasets100 = datasets.copy()
    for poe in poes:
        poe_data = datasets[poe]
        print(poe_data.shape)
        new_data = np.zeros((poe_data.shape[0], 100, poe_data.shape[2]))
        for i in range(poe_data.shape[0]):
            max_idx = np.where(poe_data[i, :, 0] < -900)[0][0]
            rate = max_idx / 100
            new_data[i, ...] = resample(poe_data[i, :max_idx, :], rate)[:100, ...]
        datasets100[poe] = new_data

    print(datasets['trunk'].shape)
    print(datasets100['trunk'].shape)

    if args.save_numpy:
        for poe in poes:
            np.save(os.path.join(args.filepath, poe) + '.npy', datasets[poe])
            np.save(os.path.join(args.filepath, poe) + '-100.npy', datasets100[poe])

    return datasets, datasets100


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filepath', default='')
    parser.add_argument('--debug', type=str2bool, nargs='?', default=False)
    parser.add_argument('--rate', type=int, default=25)
    parser.add_argument('--save_numpy', type=str2bool, help='Whether to save datasets to numpy file or return as ndarrays', default=True)
    args = parser.parse_args()

    for name in os.listdir(args.filepath):
        if '.npy' in name and 'SLS' in name:
            break

    data = np.load(os.path.join(args.filepath, name))
    # idx ???
    fps = int(name.split('.')[0].split('-')[3])

    main(args, data, fps)
