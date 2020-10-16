import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def outliers(poses):
    frames = poses.shape[0]

    for pose in np.moveaxis(poses, 0, -1):
        # print(pose.shape)
        for coords in pose:
            # print(coords.shape)
            for i in range(1, frames - 1):
                # if abs(coords[i - 1] - coords[i]) > 3:
                if abs(coords[i - 1] - coords[i]) > 0.01:
                    coords[i] = (coords[i - 1] + coords[i]) / 2

    return poses


def smoothing(poses):
    n = 2
    frames = poses.shape[0]
    part = np.ones(n + 1)
    filter = np.convolve(part, np.flip(part))
    filter = filter / np.sum(filter)
    ###
    # n = 1
    # filter = np.array([1, 5, 1])
    # filter = filter / np.sum(filter)

    for pose in np.moveaxis(poses, 0, -1):
        # print(pose.shape)
        for coords in pose:
            # print(coords.shape)
            for i in range(n, frames - n):
                coords[i] = np.sum(coords[i - n:i + n + 1] * filter)

    return poses


def fix_hip(poses):
    m = np.mean(abs(poses[:, 12, 1] - [poses[:, 11, 1]]))

    for i in range(poses.shape[0]):
        if abs(poses[i, 12, 1] - poses[i, 11, 1] < m - 10):
            mhip = (poses[i, 12, 1] + poses[i, 11, 1]) / 2
            poses[i, 11, 1] = mhip + m / 2 - 5
            poses[i, 12, 1] = mhip - m / 2 + 5

    return poses


def move_and_normalize(poses):
    poses[:, :, 0] = poses[:, :, 0] - poses[:, :, 0].min()
    poses[:, :, 1] = poses[:, :, 1] - poses[:, :, 1].min()

    poses[:, :, 0] = poses[:, :, 0] / poses[:, :, 1].max()
    poses[:, :, 1] = poses[:, :, 1] / poses[:, :, 1].max()

    return poses


def main():
    parser = ArgumentParser()
    parser.add_argument('file', help='Numpy file to filter')
    args = parser.parse_args()
    poses = np.load(args.file)
    poses = poses[5:-1, :, 0:2]

    if poses[0, 0, 0] < 1:
        poses[..., 0] = poses[..., 0] * 1920
        poses[..., 1] = poses[..., 1] * 1080

    plt.plot(poses[:, 14, 1])
    poses2 = outliers(poses)
    plt.plot(poses2[:, 14, 1])
    poses3 = smoothing(poses2)
    plt.plot(poses3[:, 14, 1])
    plt.show()


if __name__ == '__main__':
    main()
