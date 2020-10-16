import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
# import filt
from scipy.signal import find_peaks


def split(poses, debug=False):
    joint = 13
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

        # if (peaks[i] - removed) > 30 and (idx - peaks[i]) > 30:
        if seqs is None:
            seqs = np.array(poses[:idx, ...], dtype=object)
        else:
            seqs = np.array([seqs, poses[:idx, ...]], dtype=object)
            peaks[i] -= removed

        poses = poses[idx:, ...]
        removed += idx

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

        if len(seqs.shape) == 1:
            ts = seqs[1]
        else:
            ts = seqs
        print('min len {0}'.format(min_len))
        if debug:
            plt.plot(ts[:, joint, 1])
            plt.show()
            plt.plot(ts[:, joint, 1])
        print('first if: {0}, {1} and {2}, {3}'.format(peaks[-i - 1],
                                                       int(min_len / 2),
                                                       peaks[-i - 1],
                                                       len(ts) - int(min_len / 2)))
        if peaks[-i - 1] < int(min_len / 2):
            pose_seqs[i, ...] = ts[:min_len, ...]
        elif peaks[-i - 1] <= (len(ts) - int(min_len / 2)):
            print('IN IF')
            pose_seqs[i, ...] = ts[peaks[-i - 1] - int(min_len / 2):
                                   peaks[-i - 1] +
                                   int(np.ceil(min_len / 2)), ...]
        else:
            pose_seqs[i, ...] = ts[-min_len:, ...]
        seqs = seqs[0]

        # else:
        #     if debug:
        #         plt.plot(seqs[:, joint, 1])
        #         plt.show()
        #     if ((peaks[-i - 1] >= int(min_len / 2)) and
        #             (peaks[-i - 1] <= len(seqs) - int(min_len / 2))):
        #         pose_seqs[i, ...] = seqs[peaks[-i - 1] - int(min_len / 2):
        #                                  peaks[-i - 1] +
        #                                  int(np.ceil(min_len / 2)), ...]
        #     elif peaks[-i - 1] < len(seqs) / 2:
        #         pose_seqs[i, ...] = seqs[:min_len, ...]
        #     else:
        #         pose_seqs[i, ...] = seqs[-min_len:, ...]
        #
        if debug:
            plt.plot(pose_seqs[i, :, joint, 1])
            plt.show()

    print(pose_seqs[:, :, joint, 1].shape)
    # plt.plot(pose_seqs[0, :, joint, 1])
    # # plt.plot(pose_seqs[1, :, joint, 1])
    # plt.plot(pose_seqs[2, :, joint, 1])
    # plt.plot(pose_seqs[3, :, joint, 1])
    # print(pose_seqs[0, :, joint, 1])
    # print(pose_seqs[1, :, joint, 1])
    # print(pose_seqs[2, :, joint, 1])
    # print(pose_seqs[3, :, joint, 1])

    if debug:
        plt.plot(pose_seqs[:, :, joint, 1].T)
        plt.show()

    return pose_seqs, peaks


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    args = parser.parse_args()
    poses = np.load(args.np_file)

    poses = poses[6:-1, :, 0:2]

    split(poses)


if __name__ == '__main__':
    main()
