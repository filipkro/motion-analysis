import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
import numpy as np


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                        for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def main(args):
    big_data = np.load('/home/filipkr/Documents/xjob/pose-data/big_data-synced.npz',
                       allow_pickle=True)['data'].item()

    action = 'SLS1R'
    input = big_data[args.subject][action]['positions_2d'][0]
    preds = np.load(args.preds)
    truth = big_data[args.subject][action]['positions_3d'][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    vid_frame = 300

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(preds[vid_frame, :, 0], preds[vid_frame, :, 1],
               preds[vid_frame, :, 2], 'b')

    axisEqual3D(ax)
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.set_zlabel('Z / m')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    vid_frame = 600
    ax.scatter(truth[vid_frame, :, 0], truth[vid_frame, :, 1],
               truth[vid_frame, :, 2], 'b')

    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.set_zlabel('Z / m')
    axisEqual3D(ax)
    plt.show()
    # ax.plot(rotated[vid_frame, [1, 0, 4], 0], rotated[vid_frame, [1, 0, 4], 1],
    #         rotated[vid_frame, [1, 0, 4], 2], 'b')
    # ax.plot(rotated[vid_frame, [1, 2, 3], 0], rotated[vid_frame, [1, 2, 3], 1],
    #         rotated[vid_frame, [1, 2, 3], 2], 'b')
    # ax.plot(rotated[vid_frame, [4, 5, 6], 0], rotated[vid_frame, [4, 5, 6], 1],
    #         rotated[vid_frame, [4, 5, 6], 2], 'b')
    # ax.plot(rotated[vid_frame, [0, 7, 8, 9, 10], 0],
    #         rotated[vid_frame, [0, 7, 8, 9, 10], 1],
    #         rotated[vid_frame, [0, 7, 8, 9, 10], 2], 'b')
    # ax.plot(rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 0],
    #         rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 1],
    #         rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 2], 'b')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('preds')
    args = parser.parse_args()
    main(args)
