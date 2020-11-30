import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.interpolate import interp1d
import os
from create_mts import parse_mts
from post_process import filt
import re

BASELINE_FPS = 30


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def main():
    parser = ArgumentParser()
    parser.add_argument('root', help='Folder ')
    parser.add_argument('--label', type=int, default=0)
    args = parser.parse_args()
    label = args.label
    print(args.root)
    if args.root[-1] != '/':
        args.root += '/'
    print(args.root)
    mts = None
    mts_labels = None
    kpts = []
    angles = np.array([[12, 14], [14, 16]])
    i = 0
    debug = False
    for file in os.listdir(args.root):
        if file.endswith('.npy'):
            filename = args.root + file
            print(file)
            # fps = int(file[9:11])
            fps = int(re.search(r'\d+', file[7:]).group())
            # print(fps)
            poses = np.load(filename)
            poses = poses[5:, :, 0:2]
            ratio = float(fps / BASELINE_FPS)
            poses = filt.move_and_normalize(poses)
            # print(ratio)
            print(poses.shape)
            poses = resample(poses, ratio)
            # print(poses.shape)

            debug = file == 'vis_021_FL_R25HRNetTopDownCocoDataset.npy'

            if not debug:
                mts, mts_labels = parse_mts(poses, kpts.copy(), label, mts,
                                            mts_labels, angles, debug=debug)
            # mts = mts[..., :-2]
            i += 1

    print('Shape of MTS: {0}'.format(mts.shape))
    print('shape of labels {0}'.format(mts_labels.shape))
    plt.plot(mts[:, :, -1].T)
    # plt.plot(mts[1, :, 0], label='0')
    # plt.plot(mts[1, :, 1], label='1')
    # plt.plot(mts[1, :, 2], label='2')
    # plt.plot(mts[1, :, 3], label='3')
    plt.legend()
    plt.show()

    mts = mts[:, :, 1:]
    print(mts.shape)
    save_file = args.root + 'data.npz'
    np.savez(save_file, mts=mts, labels=mts_labels)


if __name__ == '__main__':
    main()
