import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
# import filt
from scipy.signal import find_peaks
import os
from scipy.interpolate import interp1d


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def normalize_coords(poses):
    dist = abs(poses[10, 0, 1] - poses[10, 16, 1])
    poses = poses / dist
    poses = poses - poses[0, 12, :]  # np.expand_dims(poses[0, 0, :], 1)
    return poses


def split_peaks(poses, joint=13, debug=False):
    # joint = 13
    peaks, _ = find_peaks(poses[:, joint, 1], distance=50, prominence=0.05,
                          width=13)

    nbr_peaks = len(peaks)
    print(nbr_peaks)
    print(peaks)
    if debug:
        plt.plot(poses[:, joint, 1])
        plt.show()

    min_len = poses.shape[0]
    removed = 0
    seqs = None
    for i in range(nbr_peaks - 1):
        idx = int(np.mean((peaks[i], peaks[i + 1]))) - removed
        min_len = np.min((min_len, idx))
        if debug:
            print('idx', idx)
            print(seqs.shape) if seqs is not None else print('none')
            plt.plot(poses[:idx, joint, 1])
            plt.show()
        # if (peaks[i] - removed) > 30 and (idx - peaks[i]) > 30:
        if seqs is None:
            seqs = np.array(poses[:idx, ...], dtype=object)
        else:
            seqs = np.array([seqs, poses[:idx, ...]], dtype=object)
            peaks[i] -= removed

        poses = poses[idx:, ...]
        removed += idx

    if debug:
        plt.plot(poses[:, joint, 1])
        plt.show()
    peaks[-1] -= removed
    seqs = np.array((seqs, poses), dtype=object)
    min_len = np.min((min_len, poses.shape[0]))

    pose_seqs = np.zeros((nbr_peaks, min_len, poses.shape[1], poses.shape[2]))

    # print(int(min_len / 2))
    print(peaks)
    for i in range(nbr_peaks):
        print(i, peaks[-i - 1])

        # if debug:
        #     plt.plot(pose_seqs[i, :, joint, 1])
        #     plt.show()
        print(seqs.shape)
        if len(seqs.shape) == 1:
            ts = seqs[1]
        else:
            ts = seqs
        print('min len {0}'.format(min_len))
        print(ts.shape)
        if debug:
            plt.plot(ts[:, joint, 1])
            plt.show()
            plt.plot(ts[:, joint, 1])
        print('first if: {0}, {1} and {2}, {3}'.format(peaks[-i - 1],
                                                       int(min_len / 2),
                                                       peaks[-i - 1],
                                                       len(ts) -
                                                       int(min_len / 2)))
        if peaks[-i - 1] <= int(min_len / 2):
            pose_seqs[i, ...] = ts[:min_len, ...]
        elif peaks[-i - 1] <= (len(ts) - int(min_len / 2)):
            print('IN IF')
            print(int(np.ceil(min_len / 2)))
            print(len(ts[peaks[-i - 1] - int(np.ceil(min_len / 2)):
                         peaks[-i - 1] - int(np.ceil(min_len / 2))
                         + min_len, ...]))
            pose_seqs[i, ...] = ts[peaks[-i - 1] - int(np.ceil(min_len / 2)):
                                   peaks[-i - 1] - int(np.ceil(min_len / 2))
                                   + min_len, ...]
        else:
            pose_seqs[i, ...] = ts[-min_len:, ...]
        seqs = seqs[0]

        if debug:
            plt.plot(pose_seqs[i, :, joint, 1])
            plt.show()

    print(pose_seqs[:, :, joint, 1].shape)

    if debug:
        plt.plot(pose_seqs[:, :, joint, 1].T)
        plt.show()
    # plt.plot(pose_seqs[:, :, joint, 1].T)
    # plt.show()
    return pose_seqs, peaks


def split_peaks_pad(poses, rate, xtra_samp=100, joint=5, debug=False,
                    only_peaks=False, prom=0.048):
    dist = int(rate * 2 / 3)
    width = int(rate / 5)

    # peaks, props = find_peaks(poses[:, joint, 1], distance=dist, prominence=0.02,
    #                           width=width, plateau_size=1)
    peaks, props = find_peaks(poses[:, joint, 1], distance=dist, prominence=prom,
                              width=width, plateau_size=1)

    nbr_peaks = len(peaks)

    left_ips = props['left_ips'].astype(int)
    right_ips = props['right_ips'].astype(int)
    peaks = ((left_ips + right_ips) / 2).astype(int)

    if only_peaks:
        edges = np.array([left_ips, right_ips])
        return peaks, edges

    if debug:
        print(props)
        print(nbr_peaks)
        print(peaks)
        print(left_ips)
        print(right_ips)

        plt.plot(poses[:, joint, 1])
        plt.show()

    # removed = 0
    idx = 0
    pose_split = np.zeros((nbr_peaks, 2 * xtra_samp,
                           poses.shape[1], poses.shape[2]))
    for i in range(nbr_peaks):
        poses = poses[idx:, ...]
        peaks -= idx
        left_ips -= idx
        right_ips -= idx
        # removed += idx
        if i < nbr_peaks - 1:
            idx = int(np.mean((right_ips[i], left_ips[i + 1])))  # - removed
        else:
            idx = poses.shape[0]

        if idx < 2 * xtra_samp:
            start_idx = 0
            end_idx = idx
        else:
            start_idx = peaks[i] - xtra_samp if peaks[i] > xtra_samp else 0
            end_idx = peaks[i] + \
                xtra_samp if peaks[i] < idx - xtra_samp else idx

        pad = 2 * xtra_samp - (end_idx - start_idx)
        pose_split[i, ...] = np.pad(poses[start_idx:end_idx, ...],
                                    ((0, pad), (0, 0), (0, 0)), 'constant',
                                    constant_values=-1000)
    # if debug:
        # plt.plot(pose_split[:, :, joint, 1].T)
        # plt.ylim((0, 0.2))
        # # plt.show()

    return pose_split, peaks


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    args = parser.parse_args()
    poses = np.load(args.np_file)

    poses = poses[6:-1, :, 0:2]

    fps = int(os.path.basename(args.np_file).split('.')[0].split('-')[3])

    rate = 25
    poses = resample(poses, fps / rate)
    poses = normalize_coords(poses)
    print(np.max(poses))
    print(np.min(poses))
    print(fps)
    samples = int(np.round(4 * fps))
    print(samples)

    # samples = 210

    split_peaks_pad(poses, rate, xtra_samp=samples, joint=5, debug=True)


if __name__ == '__main__':
    main()
