import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file1', help='First numpy file')
    parser.add_argument('np_file2', help='Second numpy file')
    args = parser.parse_args()

    poses1 = np.load(args.np_file1)
    poses1 = poses1[6:-1, :, 0:2]
    poses2 = np.load(args.np_file2)
    poses2 = poses2[6:-1, :, 0:2]

    print(np.mean(poses1[:, 0:17, :] - poses2[:, 0:17, :]))


if __name__ == '__main__':
    main()
