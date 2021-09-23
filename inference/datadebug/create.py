import numpy as np
import pickle
# import os


POE = ['hip', 'trunk', 'femval', 'kmfp']
IDXS = {'hip': [424, 425, 426, 427, 428], 'trunk': [424, 425, 426, 427, 428],
        'femval': [434, 435, 436, 437, 438], 'kmfp': [434, 435, 436, 437, 438]}
LITS = {'hip': 'Sigrid-Undset', 'trunk': 'Isaac-Bashevis-Singer',
        'femval': 'Olga-Tokarczuk', 'kmfp': 'Mikhail-Sholokhov'}
BASE = '/home/filipkr/Documents/xjob/data/datasets/data_'
OUT = '/home/filipkr/Documents/xjob/app-mm/inference/datadebug/'
datasets = {'femval': [], 'trunk': [], 'hip': [], 'kmfp': []}
datasets100 = datasets.copy()


def main():

    for poe in POE:
        idx = IDXS[poe]
        lit = LITS[poe]

        x = np.load(BASE + lit + '.npz')
        x100 = np.load(BASE + lit + '_len100.npz')

        x = x['mts']
        x100 = x100['mts']

        print(x.shape)
        print(x100.shape)

        x = x[idx, ...]
        x100 = x100[idx, ...]

        print(x.shape)
        print(x100.shape)

        datasets[poe] = x
        datasets100[poe] = x100

    f = open(OUT + 'data.pkl', 'wb')
    pickle.dump(datasets, f)
    f.close()
    f = open(OUT + 'data100.pkl', 'wb')
    pickle.dump(datasets100, f)
    f.close()


if __name__ == '__main__':
    main()
