import os
import numpy as np
from argparse import ArgumentParser

marked_beg = {'01-SLS-L-25.npy': 720, '41-SLS-L-60.npy': 480,
              '44-SLS-R-60.npy': 480, '45-SLS-R-60.npy': 480,
              '28-SLS-R-60.npy': 240, '52-SLS-R-60.npy': 620}

marked_end = {'04-SLS-L-25.npy': 510, '37-SLS-L-60.npy': 1400,
              '25-SLS-R-60.npy': 1560, '47-SLS-L-60.npy': 1440}

ttb_end = {'03-SLS-R-30.npy': 540}
healthy_end = {'01-SLS-R-25.npy': 800}


def main(args):
    dataset = os.path.basename(args.root).split('-')[0]
    crop_end = {}
    crop_beg = {}
    if dataset == 'marked':
        crop_end = marked_end
        crop_beg = marked_beg
    elif dataset == 'ttb':
        crop_end = ttb_end
    elif dataset == 'healthy':
        crop_end = healthy_end
    else:
        return

    for file in os.listdir(args.root):
        if file.endswith('.npy'):
            fp = os.path.join(args.root, file)
            data = np.load(fp)

            if file in crop_beg.keys():
                data = data[crop_beg[file]:,...]
                np.save(fp, data)
                print('new save: {}'.format(fp))
            if file in crop_end.keys():
                data = data[:crop_end[file],...]
                np.save(fp, data)
                print('new save: {}'.format(fp))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    main(args)
