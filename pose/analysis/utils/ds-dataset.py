import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))

def main(args):
    data = np.load(args.data)
    new_path = args.data.split('.npz')[0] + '_ds.npz'
    x = data['mts']

    new_x = np.zeros((x.shape[0], int(x.shape[1]/2.5), x.shape[2]))
    # print(new_x.shape)
    # print(x.shape)
    for i, entry in enumerate(x):
        max_idx = np.where(entry[:, 0] < -900)[0][0]
        downsampled = resample(entry[:max_idx, ...], 2.5)
        new_x[i, :downsampled.shape[0], :] = downsampled
        new_x[i, downsampled.shape[0]:, :] = -1000 * np.ones(new_x[i, downsampled.shape[0]:, :].shape)

    np.savez(new_path, mts=new_x, labels=data['labels'])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    main(args)
