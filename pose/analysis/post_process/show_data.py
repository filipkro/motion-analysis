import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import filt
# from findpeaks import findpeaks
from scipy.signal import find_peaks


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    args = parser.parse_args()
    poses = np.load(args.np_file)

    poses = poses[6:-1, :, 0:2]

    # if poses[0, 0, 0] < 1:
    #     poses[..., 0] = poses[..., 0] * 1920
    #     poses[..., 1] = poses[..., 1] * 1080
    # poses = filt.outliers(poses)
    # poses = filt.smoothing(poses)
    # poses = filt.fix_hip(poses)

    poses_norm = np.copy(poses)

    poses_norm = filt.move_and_normalize(poses_norm)

    # print(poses_pre_process - poses)
    # plt.plot(poses_pre_process[:, 14, 0])
    plt.figure(1)
    plt.plot(poses[:, 14, 0], label='orig x')
    plt.plot(poses[:, 14, 1], label='orig y')

    plt.plot(poses_norm[:, 14, 0], label='normalized x')
    plt.plot(poses_norm[:, 14, 1], label='normalized y')
    plt.legend()
    plt.show()

    # fp = findpeaks(method='peakdetect')
    # fp = findpeaks(method='topology', limit=1)
    peaks, props = find_peaks(poses_norm[:, 14, 1], distance=40, width=20,
                              prominence=None)
    print(peaks)

    i1 = int(np.mean(peaks[0:2]))
    i2 = int(np.mean(peaks[1:3]))

    seq1 = poses_norm[:i1, ...]
    seq2 = poses_norm[i1:i2, ...]
    seq3 = poses_norm[i2:, ...]

    seq_len = min((len(seq1), len(seq2), len(seq3)))
    print(seq_len)
    seq3 = seq3[-seq_len:] if peaks[2] > len(seq1) + \
        len(seq2) + len(seq1) / 2 else seq3[:seq_len]
    seq1 = seq1[-seq_len:] if peaks[0] > len(seq1) / 2 else seq1[:seq_len]

    plt.figure(2)
    plt.plot(seq1[:, 14, 1], label='1st, y')
    plt.plot(seq2[:, 14, 1], label='2nd, y')
    plt.plot(seq3[:, 14, 1], label='3rd, y')
    # plt.plot(poses_norm[:i1,14,1],label='1st, y')
    # for i in range(len(peaks) - 1):

    # fp.plot1d()
    # plt.plot(poses[:, 13, 0], label='left x')
    # plt.plot(poses[:, 13, 1], label='left y')
    plt.legend()
    plt.show()
    # dhip = abs(poses[:, 11, 1] - poses[:, 12, 1])
    # hip_mean = np.mean(dhip) * np.ones(dhip.shape)
    # hip_med = np.median(dhip) + np.ones(dhip.shape)
    # plt.plot(dhip)
    # plt.plot(hip_mean)
    # plt.plot(hip_med)
    # plt.show()


if __name__ == '__main__':
    main()
