import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def calc_angle(poses, kpts):
    return np.arctan2(poses[:, kpts[0], 1] -
                      poses[:, kpts[1], 1],
                      poses[:, kpts[0], 0] -
                      poses[:, kpts[1], 0])


# def main(args):
#     data = np.load(args.file)
#     plt.plot(data[:, 5, 1])
#     plt.figure()
#     angs = calc_angle(data, [12, 14])
#     plt.plot(angs)
#     plt.show()

def main(args):
    data = np.load(args.file)['mts']
    print(data.shape)
    nbr_plots = data.shape[2]
    fig, axs = plt.subplots(nbr_plots,gridspec_kw={'hspace':0.05})
    fig.suptitle('Multivariate Time Series', fontsize=18)
    for i in range(nbr_plots):
        axs[i].plot(data[50,:,i], linewidth=8)
        if i < nbr_plots-1:
            axs[i].axes.xaxis.set_ticklabels([])

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    main(args)
