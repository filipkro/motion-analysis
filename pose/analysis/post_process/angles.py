import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.interpolate import interp1d


def flip_direction(head, foot):
    return abs(head[0] - foot[0]) > abs(head[1] - foot[1])


def get_angles(part1, part2, flip=False):
    if flip:
        part1 = np.moveaxis(part1, 0, -1)
        part2 = np.moveaxis(part2, 0, -1)

    angles1 = np.arctan2(part1[:, 1] - part2[:, 1], part1[:, 0] - part2[:, 0])
    angles2 = np.arctan2(part2[:, 0] - part1[:, 0], part2[:, 1] - part1[:, 1])

    for i in range(len(angles2)):
        a = angles2[i]
        # while a > np.pi:
        #     a -= 2 * np.pi
        while a < 0:
            a += 2 * np.pi
        angles2[i] = a - np.pi
    return angles1, angles2


def read_angles(file):
    angles = np.genfromtxt(file)  # , dtype=str, delimiter='\t')
    print(angles.shape)
    print(angles.dtype)

    return angles


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.size / factor))
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    parser.add_argument('--angle_file', default='',
                        help='tab separated file containing angles')
    args = parser.parse_args()
    poses = np.load(args.np_file)
    poses = poses[6:-1, :, 0:2]

    flip = flip_direction(poses[0, 0, :], poses[0, 16, :])

    angles1, angles2 = get_angles(poses[:, 14, :], poses[:, 12, :], flip)
    # print(angles)

    angles1 = angles1 * 180 / np.pi - 90  # + 1.5
    angles2 = angles2 * 180 / np.pi + 3.5

    if args.angle_file != '':
        angles_from_file = read_angles(args.angle_file)
        # angles_from_file = angles_from_file[500:-1]
        # angles1 = angles1[180:-1]
        # angles_from_file = angles_from_file[500:-1]
        # ratio = len(angles_from_file[:, 2]) / len(angles1)
        # angles_ds = resample(angles_from_file[:, 2], ratio)
        # plt.figure(2)
        # # plt.plot(angles_from_file[:, 1], label='1')
        # plt.plot(angles_from_file[:, 2], label='2')
        # # plt.plot(angles_from_file[:, 3], label='3')
        # plt.legend()
        # plt.figure(3)
        # plt.plot(angles_from_file[:, 4], label='4')
        # plt.plot(angles_from_file[:, 5], label='5')
        # plt.plot(angles_from_file[:, 6], label='6')
        # plt.legend()

        # plt.plot(angles_from_file[:, 8], label='8')
        # plt.plot(angles_from_file[:, 9], label='9')
        # plt.legend()
        # angles_from_file = angles_from_file[500:-1]
        # angles1 = angles1[180:-1]
        ratio = len(angles_from_file[:, 2]) / len(angles1)

        plt.figure(3)
        ang = resample(angles_from_file[:, 7], ratio)
        plt.plot(ang, label='7')
        plt.plot(angles2, label='test')
        plt.plot()

        angles_ds = resample(angles_from_file[:, 2], ratio)
        print('len ds {0}\nlen orig {1}'.format(len(angles_ds), len(angles1)))
        plt.figure(4)
        # plt.plot(angles1 - angles_ds)
        plt.plot(angles_ds, label='Measured')
        plt.plot(angles1, label='Estimated')
        # plt.plot(angles2*180/np.pi, label='a2')
        plt.legend(prop={'size': 17})
        plt.xlabel('Frames', size=17)
        plt.ylabel('Angle / deg', size=17)

    # angles2 = angles2  # * 180 / np.pi
    # plt.figure(1)
    # plt.plot(angles1)
    # plt.plot(angles2)
    # plt.figure(2)

    plt.show()


if __name__ == '__main__':
    main()
