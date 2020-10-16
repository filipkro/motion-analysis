import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def lin_interpol(poses):
    rate = 1
    for i in range(1, poses.shape[0]):
        if poses[i, 0, 0] == 0:
            rate += 1
        else:
            break

    for i in range(1, poses.shape[0] - rate, rate):
        poses[i:i + rate - 1, ...] = poses[i - 1, ...] * \
            np.ones(poses[i:i + rate - 1, ...].shape)

        diff = poses[i + rate - 1, ...] - poses[i - 1, ...] \
            * np.ones(poses[i:i + rate - 1, ...].shape)
        interpolation = np.repeat(np.arange(1, rate)[
            :, np.newaxis] / rate, diff.shape[1], axis=1)
        interpolation = np.repeat(interpolation[:, :, np.newaxis], 2, axis=2)

        poses[i:i + rate - 1, ...] += diff * interpolation

    poses = poses[0:i + rate, ...]

    return poses


def main():
    parser = ArgumentParser()
    parser.add_argument('file', help='Numpy file to interpolate')
    args = parser.parse_args()
    poses = np.load(args.file)
    poses = poses[:, :, 0:2]
    # plt.plot(poses[:, 0, 0])
    poses_int = lin_interpol(poses)

    plt.plot(poses_int[:, 16, 0])
    plt.plot(poses_int[:, 16, 1])
    plt.show()


if __name__ == '__main__':
    main()
