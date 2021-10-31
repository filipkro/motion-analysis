import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

def crop(data, idx):
    data = data[:idx, ...]
    return data

def crop_start(data, idx):
    data = data[idx:, ...]
    return data

def calc_angle(poses, kpts):
    return np.arctan2(poses[:, kpts[0], 1] -
                      poses[:, kpts[1], 1],
                      poses[:, kpts[0], 0] -
                      poses[:, kpts[1], 0])

def main(args):
    data = np.load(args.file)
    if args.idx is None and args.s is None:
        plt.plot(data[:,5,1])
        plt.figure()
        plt.plot(calc_angle(data, [12,14]))
        plt.show()
    elif args.idx is not None:
        data = crop(data, args.idx)
        plt.plot(data[:,5,1])
        plt.figure()
        plt.plot(calc_angle(data, [12,14]))
        plt.show()

        np.save(args.file, data)
    else:
        data = crop_start(data, args.s)
        plt.plot(data[:,5,1])
        plt.figure()
        plt.plot(calc_angle(data, [12,14]))
        plt.show()

        np.save(args.file, data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--s', type=int, default=None)
    args = parser.parse_args()
    main(args)
