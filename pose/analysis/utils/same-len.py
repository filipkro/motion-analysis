import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pandas as pd
import os
from scipy.interpolate import interp1d
import scipy
import matplotlib.pyplot as plt

def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def main(args):
    data = np.load(args.file)
    new_path = args.file.split('.npz')[0] + '_len100.npz'
    x = data['mts']

    x_new = np.zeros((x.shape[0], args.len, x.shape[2]))

    for i, entry in enumerate(x):
        max_idx = np.where(entry[:, 0] < -900)[0][0]
        rate = max_idx / args.len
        x_new[i, ...] = resample(entry[:max_idx, ...], rate)[:args.len, ...]

    # plt.plot(x_new[..., 0].T)
    # plt.figure()
    # plt.plot(x[...,0].T)
    # plt.show()

    np.savez(new_path, mts=x_new, labels=data['labels'])
    print('--DONE--')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--len', type=int, default=100)

    args = parser.parse_args()

    main(args)
