import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.interpolate import interp1d
import os
from create_mts import parse_mts
from post_process import filt

BASELINE_FPS = 30


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def main():
    parser = ArgumentParser()
    parser.add_argument('root', help='Folder ')
    args = parser.parse_args()
    print(args.root)
    if args.root[-1] != '/':
        args.root += '/'
    print(args.root)
    mts = None
    mts_labels = None
    kpts = [12, 13]
    angles = np.array([[12, 14]])
    i = 0
    for file in os.listdir(args.root):
        if file.endswith('.npy'):
            filename = args.root + file
            print(file)
            fps = int(file[9:11])
            # print(fps)
            poses = np.load(filename)
            poses = poses[5:, :, 0:2]
            ratio = float(fps / BASELINE_FPS)
            poses = filt.move_and_normalize(poses)
            # print(ratio)
            print(poses.shape)
            poses = resample(poses, ratio)
            # print(poses.shape)

            # debug = file == 'vis_002FL25HRNetTopDownCocoDataset.npy'
            debug = False
            mts, mts_labels = parse_mts(poses, kpts.copy(), i % 2, mts,
                                        mts_labels, angles, debug=debug)
            # mts = mts[..., :-2]
            i += 1

    print('Shape of MTS: {0}'.format(mts.shape))
    print('shape of labels {0}'.format(mts_labels.shape))
    plt.plot(mts[:, :, -1].T)
    plt.show()

    save_file = args.root + 'data.npz'
    np.savez(save_file, mts=mts, labels=mts_labels)


if __name__ == '__main__':
    main()
