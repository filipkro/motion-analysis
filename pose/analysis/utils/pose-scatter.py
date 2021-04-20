import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def main(args):
    data = np.load(args.file)

    # plt.scatter(data[10,[15,17, 18, 19],0], -data[10,[15,17, 18, 19],1])
    plt.scatter(data[10,16,0], -data[10,16,1], label='16')
    plt.scatter(data[10,20,0], -data[10,20,1], label='20')
    plt.scatter(data[10,21,0], -data[10,21,1], label='21')
    plt.scatter(data[10,22,0], -data[10,22,1], label='22')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    main(args)
