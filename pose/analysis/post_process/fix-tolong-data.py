import os
import numpy as np
from argparse import ArgumentParser

def main(args):
    for file in os.listdir(args.root):
        if file.endswith('.npy'):
            fp = os.path.join(args.root, file)
            data = np.load(fp)
            idx  = np.where(data[:,5,1] == 0)
            if len(idx[0]) > 0:
                # print(idx[0][0])
                # print(data.shape)
                data = data[:idx[0][0],...]
                # print(data.shape)
                np.save(fp, data)
                print('new save: {}'.format(fp))
            # print(len(idx[0]))
            # return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    main(args)
