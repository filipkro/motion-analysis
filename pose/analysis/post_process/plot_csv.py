import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.interpolate import interp1d


def rescale(poses, im=False):
    if im:
        dist = abs(poses[5, 12, 1] - poses[5, 16, 1])
    else:
        dist = abs(poses[25, 2] - poses[25, -1])

    print(dist)

    return poses / dist, dist


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.size / factor))
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))


def main():
    parser = ArgumentParser()
    parser.add_argument('csv_file', help='File')
    parser.add_argument('np_file', default='', help='Np file to compare with')
    args = parser.parse_args()

    poses_csv = np.genfromtxt(args.csv_file, delimiter=',')
    # print(poses.shape)
    # print(poses[0, :])
    # print(poses[1, :])
    # print(poses[2, :])
    coords = [2, 3, 4, 7, 8, 9, 12, 13, 14]
    poses = poses_csv[2:, coords]
    poses, dist = rescale(poses)

    rh = poses[2:, 2:5]
    print(rh.shape)

    rk = poses[2:, 7:10]
    print(rk.shape)

    ra = poses[2:, 12:]
    print(ra.shape)

    fig, axs = plt.subplots(2)
    # axs[0].plot(poses[:, 0], label='X')
    axs[0].plot(-poses[:, 2], label='Z')
    # axs[0].plot(rh[:, 0], label='X')
    # # axs[0].legend()
    # axs[0].plot(-rh[:, 2], label='Z')
    # axs[0].plot(-ra[:, 2], label='ankle')
    # axs[0].plot(-rk[:, 2], label='knee')
    axs[0].legend()

    if args.np_file != '':
        meas = np.load(args.np_file)
        meas, _ = rescale(meas, True)
        # axs[1].plot(meas[:, 12, 0], label='Measured X')
        diff = -poses[25, 2] - meas[5, 12, 1]
        axs[1].plot(meas[:, 12, 1] + diff, label='Measured Y')
        ratio = len(poses[:, 2]) / len(meas[:, 12, 1])
        print('ratio {0} \n'.format(ratio))
        ds = resample(-poses[:, 2], ratio)
        axs[1].plot(ds, label='Markers')
        axs[1].legend()

        plt.figure(2)
        plt.plot(meas[:, 12, 1] + diff, label='Estimated Y')
        plt.plot(ds, label='Markers')
        plt.legend()

        e = (meas[:, 12, 1] + diff - ds) * dist
        m = np.mean(e)
        print(m)
        print(np.std(e - m))
        print(np.max(abs(e - m)))

        plt.figure(3)
        plt.plot(e)
        plt.ylabel('Error / m')

        ds_k = resample(-poses[:, 5], ratio)

        plt.figure(4)
        plt.plot(ds_k)
        plt.plot(meas[:, 14, 1] + diff)

        ds_a = resample(-poses[:, 8], ratio)

        plt.figure(5)
        plt.plot(ds_a)
        plt.plot(meas[:, 16, 1] + diff)

    plt.show()


if __name__ == '__main__':
    main()
