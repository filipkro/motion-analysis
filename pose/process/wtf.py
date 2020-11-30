import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import pandas as pd
from scipy.interpolate import interp1d


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def read_csv(subject, action):
    root = '/home/filipkr/Documents/xjob/pose-data/'
    file = root + subject + 'RDLC_TEST.csv'
    field = subject + action
    df = pd.read_csv(file, delimiter='\t')
    data = df.filter(like=field).values
    poses = data[4:, -24:].astype('float32')
    poses = np.reshape(poses, (poses.shape[0], 8, 3))
    orig = poses.copy()
    tmp = poses[:, :, 1].copy()
    poses[:, :, 1] = -poses[:, :, 2]
    poses[:, :, 2] = tmp
    ratio = 150 / 60
    poses = resample(poses, ratio)

    center_hip = (poses[:, 0, :] + poses[:, 3, :]) / 2
    poses = np.insert(poses, 0, center_hip, axis=1)

    poses = fix_3d(poses)

    viz(poses)
    viz_orig(orig)
    plt.show()

    return poses


def viz(poses, orig=False):
    plt.figure()
    plt.scatter(poses[100, :, 0],
                poses[100, :, 1])

    plt.plot(poses[100, [1, 0, 4], 0],
             poses[100, [1, 0, 4], 1])
    plt.plot(poses[100, [1, 2, 3], 0],
             poses[100, [1, 2, 3], 1])
    plt.plot(poses[100, [0, 6, 5], 0],
             poses[100, [0, 6, 5], 1])
    plt.plot(poses[100, [8, 5, 7], 0],
             poses[100, [8, 5, 7], 1])


def viz_orig(poses):
    plt.figure()
    plt.scatter(poses[100, :, 0],
                -poses[250, :, 2])

    # plt.plot(poses[100, [1, 0, 4], 0],
    #          -poses[100, [1, 0, 4], 2])
    # plt.plot(poses[100, [1, 2, 3], 0],
    #          -poses[100, [1, 2, 3], 2])
    # plt.plot(poses[100, [0, 6, 5], 0],
    #          -poses[100, [0, 6, 5], 2])
    # plt.plot(poses[100, [8, 5, 7], 0],
    #          -poses[100, [8, 5, 7], 2])


def fix_2d(poses):

    dx = np.max(poses[:, :, 0]) - np.min(poses[:, :, 0])
    dy = np.max(poses[:, :, 1]) - np.min(poses[:, :, 1])
    norm_dist = np.max((dx, dy))
    # print(norm_dist)
    poses = 2 * poses / norm_dist - 1

    shift_y = (np.max(poses[:, :, 1]) + np.min(poses[:, :, 1])) / 2
    poses[..., 1] = poses[..., 1] - shift_y
    shift_x = (np.max(poses[:, :, 0]) + np.min(poses[:, :, 0])) / 2
    poses[..., 0] = poses[..., 0] - shift_x

    return poses


def fix_3d(poses):

    dx = np.max(poses[:, :, 0]) - np.min(poses[:, :, 0])
    dy = np.max(poses[:, :, 1]) - np.min(poses[:, :, 1])
    dz = np.max(poses[:, :, 2]) - np.min(poses[:, :, 2])
    norm_dist = np.max((dx, dy, dz))
    # print(norm_dist)
    poses = 2 * poses / norm_dist - 1

    shift_y = (np.max(poses[:, :, 1]) + np.min(poses[:, :, 1])) / 2
    poses[..., 1] = poses[..., 1] - shift_y
    shift_x = (np.max(poses[:, :, 0]) + np.min(poses[:, :, 0])) / 2
    poses[..., 0] = poses[..., 0] - shift_x
    shift_z = (np.max(poses[:, :, 2]) + np.min(poses[:, :, 2])) / 2
    poses[..., 2] = poses[..., 2] - shift_z

    # ratio = 150 / 60
    # poses = resample(poses, ratio)
    poses = poses - np.expand_dims(poses[:, 0, :], 1)
    return poses


def main():
    big_data = np.load('/home/filipkr/Documents/xjob/pose-data/big_data-synced.npz',
                       allow_pickle=True)['data'].item()

    for subject in big_data.keys():
        for action in big_data[subject].keys():
            print(subject)
            print(big_data[subject][action]['positions_3d'])
            poses = big_data[subject][action]['positions_3d'][0]
            plt.figure()
            plt.scatter(poses[100, :, 0],
                        poses[100, :, 1])

            plt.plot(poses[100, [1, 0, 4], 0],
                     poses[100, [1, 0, 4], 1])
            plt.plot(poses[100, [1, 2, 3], 0],
                     poses[100, [1, 2, 3], 1])
            plt.plot(poses[100, [0, 6, 5], 0],
                     poses[100, [0, 6, 5], 1])
            plt.plot(poses[100, [8, 5, 7], 0],
                     poses[100, [8, 5, 7], 1])

            plt.figure()
            plt.scatter(poses[100, :, 0],
                        poses[100, :, 2])

            plt.plot(poses[100, [1, 0, 4], 0],
                     poses[100, [1, 0, 4], 2])
            plt.plot(poses[100, [1, 2, 3], 0],
                     poses[100, [1, 2, 3], 2])
            plt.plot(poses[100, [0, 6, 5], 0],
                     poses[100, [0, 6, 5], 2])
            plt.plot(poses[100, [8, 5, 7], 0],
                     poses[100, [8, 5, 7], 2])

            plt.figure()
            plt.scatter(poses[100, :, 1],
                        poses[100, :, 2])

            plt.plot(poses[100, [1, 0, 4], 1],
                     poses[100, [1, 0, 4], 2])
            plt.plot(poses[100, [1, 2, 3], 1],
                     poses[100, [1, 2, 3], 2])
            plt.plot(poses[100, [0, 6, 5], 1],
                     poses[100, [0, 6, 5], 2])
            plt.plot(poses[100, [8, 5, 7], 1],
                     poses[100, [8, 5, 7], 2])

            print(subject)
            plt.show()
            # poses2d = big_data[subject][action]['positions_2d'][0]
            # poses2d = fix_2d(poses2d)
            # big_data[subject][action]['positions_2d'][0] = poses2d
            #
            # poses3d = read_csv(subject, action)
            # # print(poses3d)
            # big_data[subject][action]['positions_3d'] = poses3d
            #
            # print(subject)

    # np.savez_compressed(
    #     '/home/filipkr/Documents/xjob/pose-data/big_data-fixed.npz',
    #     data=big_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser
    main()
