import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.interpolate import interp1d


def rescale(poses, im=False):
    if im:
        dist = abs(poses[5, 12, 1] - poses[5, 16, 1])
    else:
        dist = abs(poses[25, 2] - poses[25, 8])

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
    coords = [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19]
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
    # axs[0].plot(-poses[:, 2], label='Z')
    # axs[0].plot(rh[:, 0], label='X')
    # # axs[0].legend()
    # axs[0].plot(-rh[:, 2], label='Z')
    # axs[0].plot(-ra[:, 2], label='ankle')
    # axs[0].plot(-rk[:, 2], label='knee')
    # axs[0].legend()

    if args.np_file != '':
        meas = np.load(args.np_file)
        meas, _ = rescale(meas, True)
        diffX = poses[25, 0] - meas[5, 12, 0]

        ratio = len(poses[:, 2]) / len(meas[:, 12, 1])
        dsX = resample(poses[:, 0], ratio)

        diff = -poses[25, 2] - meas[5, 12, 1]
        axs[1].plot(dist * (meas[:, 12, 1] + diff), label='Image Y')

        print('ratio {0} \n'.format(ratio))
        ds = resample(-poses[:, 2], ratio)
        axs[1].plot(dist * ds, label='Marker - Y')
        axs[1].legend()

        axs[0].plot(dist * (meas[:, 12, 0] + diffX), label='Image - X')
        plt.title('Right hip')
        axs[0].plot(dist * dsX, label='Marker - X')
        axs[0].legend()

        eY = (meas[:, 12, 1] + diff - ds) * dist
        eX = (meas[:, 12, 0] + diffX - dsX) * dist

        # plt.figure(2)
        fig, axs = plt.subplots(2)
        diffX = poses[25, 3] - meas[5, 14, 0]

        dsX = resample(poses[:, 3], ratio)

        diff = -poses[25, 5] - meas[5, 14, 1]
        axs[1].plot(dist * (meas[:, 14, 1] + diff + 0.025), label='Image Y')

        print('ratio {0} \n'.format(ratio))
        ds = resample(-poses[:, 5], ratio)
        axs[1].plot(dist * ds, label='Marker - Y')
        axs[1].legend()

        axs[0].plot(dist * (meas[:, 14, 0] + diffX + 0.025), label='Image - X')
        plt.title('Right knee')
        axs[0].plot(dist * dsX, label='Marker - X')
        axs[0].legend()

        eKY = (meas[:, 14, 1] + diff - ds) * dist
        eKX = (meas[:, 14, 0] + diffX - dsX) * dist

        fig, axs = plt.subplots(2)
        diffX = poses[25, 6] - meas[5, 16, 0]

        dsX = resample(poses[:, 6], ratio)

        diff = -poses[25, 8] - meas[5, 16, 1]
        axs[1].plot(meas[:, 16, 1] + diff, label='Image Y')

        print('ratio {0} \n'.format(ratio))
        ds = resample(-poses[:, 8], ratio)
        axs[1].plot(ds, label='Marker - Y')
        axs[1].legend()

        axs[0].plot(meas[:, 16, 0] + diffX, label='Image - X')
        plt.title('Right ankle')
        axs[0].plot(dsX, label='Marker - X')
        axs[0].legend()

        fig, axs = plt.subplots(2)
        diffX = poses[25, 9] - meas[5, 11, 0]

        dsX = resample(poses[:, 9], ratio)

        diff = -poses[25, 11] - meas[5, 11, 1]
        axs[1].plot(meas[:, 11, 1] + diff, label='Image Y')

        print('ratio {0} \n'.format(ratio))
        ds = resample(-poses[:, 11], ratio)
        axs[1].plot(ds, label='Marker - Y')
        axs[1].legend()

        axs[0].plot(meas[:, 11, 0] + diffX, label='Image - X')
        plt.title('Right ankle')
        axs[0].plot(dsX, label='Marker - X')
        axs[0].legend()

        # plt.plot(meas[:, 12, 1] + diff, label='Estimated Y')
        # plt.plot(ds, label='Markers')
        # plt.legend()

        # m = np.mean(eY)
        m = 0
        print(m)
        print(np.std(eY - m))
        print(np.max(abs(eY - m)))

        plt.figure(5)
        plt.plot(eY, label='error y')
        plt.plot(eX, label='error x')
        plt.ylabel('Error / m (?)')
        plt.legend()
        plt.title('error hip')

        plt.figure(6)
        plt.plot(eKY, label='error y')
        plt.plot(eKX, label='error x')
        plt.ylabel('Error / m (?)')
        plt.legend()
        plt.title('error knee')

    plt.show()


if __name__ == '__main__':
    main()
