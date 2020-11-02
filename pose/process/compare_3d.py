import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from matplotlib import cm
from celluloid import Camera
from scipy.interpolate import interp1d


def extract_joints(data):
    joint_idx = np.array([[2, 3, 4], [7, 8, 9], [12, 13, 14],
                          [17, 18, 19], [22, 23, 24], [27, 28, 29],
                          [32, 33, 34], [38, 39, 40]])
    # print(data.shape)
    poses = data[2:, joint_idx]
    center_hip = (poses[:, 0, :] + poses[:, 3, :]) / 2
    poses = np.insert(poses, 0, center_hip, axis=1)
    # print(center_hip.shape)
    poses = poses - np.expand_dims(poses[:, 0, :], 1)
    return poses


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def main():
    parser = ArgumentParser()
    parser.add_argument('csv_file', help='File')
    parser.add_argument('np_file', default='', help='Np file to compare with')
    args = parser.parse_args()

    truth = np.genfromtxt(args.csv_file, delimiter=',')
    preds = np.load(args.np_file)
    dpreds = np.sqrt((np.mean(preds[:, 0, 0]) - np.mean(preds[:, 8, 0]))**2 +
                     (np.mean(preds[:, 0, 1]) - np.mean(preds[:, 8, 1]))**2 +
                     (np.mean(preds[:, 0, 2]) - np.mean(preds[:, 8, 2]))**2)

    poses = extract_joints(truth)
    dtruths = np.sqrt((np.mean(poses[:, 0, 0]) - np.mean(poses[:, 5, 0]))**2 +
                      (np.mean(poses[:, 0, 1]) - np.mean(poses[:, 5, 1]))**2 +
                      (np.mean(poses[:, 0, 2]) - np.mean(poses[:, 5, 2]))**2)
    preds = preds * dtruths / dpreds
    # truth = truth[]

    rotation_radians = -np.pi / 9
    # rotation_radians = 0
    rotation_axis = np.array([1, 0, 0])
    rotation_vector = rotation_radians * rotation_axis
    rotation = Rot.from_rotvec(rotation_vector)

    rotated = np.zeros(preds.shape)
    for i in range(preds.shape[0]):
        rotated[i, ...] = rotation.apply(preds[i, ...])

    print(np.sum(rotated - preds))

    ratio = rotated.shape[0] / poses.shape[0]
    upsampled = resample(rotated, ratio)
    print(upsampled.shape)
    print(poses.shape)

    error_rh = np.sqrt((poses[:, 1, 0] - upsampled[:, 1, 0])
                       ** 2 + (poses[:, 1, 2] + upsampled[:, 1, 1])**2
                       + (poses[:, 1, 1] - upsampled[:, 1, 2])**2)
    error_lh = np.sqrt((poses[:, 4, 0] - upsampled[:, 4, 0])
                       ** 2 + (poses[:, 4, 2] + upsampled[:, 4, 1])**2
                       + (poses[:, 4, 1] - upsampled[:, 4, 2])**2)
    error_rk = np.sqrt((poses[:, 2, 0] - upsampled[:, 2, 0])
                       ** 2 + (poses[:, 2, 2] + upsampled[:, 2, 1])**2
                       + (poses[:, 2, 1] - upsampled[:, 2, 2])**2)
    error_ra = np.sqrt((poses[:, 3, 0] - upsampled[:, 3, 0])
                       ** 2 + (poses[:, 3, 2] + upsampled[:, 3, 1])**2
                       + (poses[:, 3, 1] - upsampled[:, 3, 2])**2)
    #
    # error_rh = np.sqrt((poses[:, 1, 0] + upsampled[:, 4, 0])
    #                    ** 2 + (poses[:, 1, 2] + upsampled[:, 4, 1])**2
    #                    + (poses[:, 1, 1] + upsampled[:, 4, 2])**2)
    # error_lh = np.sqrt((poses[:, 4, 0] + upsampled[:, 1, 0])
    #                    ** 2 + (poses[:, 4, 2] + upsampled[:, 1, 1])**2
    #                    + (poses[:, 4, 1] + upsampled[:, 1, 2])**2)
    # error_rk = np.sqrt((poses[:, 2, 0] + upsampled[:, 5, 0])
    #                    ** 2 + (poses[:, 2, 2] + upsampled[:, 5, 1])**2
    #                    + (poses[:, 2, 1] + upsampled[:, 5, 2])**2)
    # error_ra = np.sqrt((poses[:, 3, 0] + upsampled[:, 6, 0])
    #                    ** 2 + (poses[:, 3, 2] + upsampled[:, 6, 1])**2
    #                    + (poses[:, 3, 1] + upsampled[:, 6, 2])**2)

    # print('rh1: {}, rh: {}, lh1: {}, lh: {}, rk1: {}, rk: {}, ra1: {}, ra {}'.format(np.linalg.norm(error_rh1), np.linalg.norm(error_rh), np.linalg.norm(error_lh1),
    #                                                                                  np.linalg.norm(error_lh), np.linalg.norm(
    #                                                                                      error_rk1), np.linalg.norm(error_rk),
    #                                                                                  np.linalg.norm(error_ra1), np.linalg.norm(error_ra)))

    # plt.figure()
    # plt.plot(error_rk1, label='error_rk1')
    # plt.plot(error_rk, label='error_rk')
    # # plt.plot((poses[:, 4, 1] - upsampled[:, 1, 2]), label='z')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot((poses[:, 2, 2]), label='1')
    # plt.plot(-upsampled[:, 3, 1], label='2')
    # # plt.plot((poses[:, 2, 2]), label='3')
    # plt.plot(-upsampled[:, 6, 1], label='4')
    # # plt.plot((poses[:, 4, 1] + upsampled[:, 1, 2]), label='z')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot((poses[:, 2, 1]), label='1')
    # plt.plot(upsampled[:, 3, 2], label='2')
    # # plt.plot((poses[:, 2, 2]), label='3')
    # plt.plot(upsampled[:, 6, 2], label='4')
    # # plt.plot((poses[:, 4, 1] + upsampled[:, 1, 2]), label='z')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot((poses[:, 2, 0]), label='1')
    # plt.plot(upsampled[:, 3, 0], label='2')
    # # plt.plot((poses[:, 2, 2]), label='3')
    # plt.plot(upsampled[:, 6, 0], label='4')
    # # plt.plot((poses[:, 4, 1] + upsampled[:, 1, 2]), label='z')
    # plt.legend()

    _, axs = plt.subplots(2, 2)
    axs[0, 0].plot(error_rh, label='Right hip')
    axs[1, 0].plot(error_lh, label='Left hip')
    axs[0, 1].plot(error_rk, label='Right knee')
    axs[1, 1].plot(error_ra, label='Right ankle')
    for a in axs:
        a[0].legend(prop={'size': 20})
        a[1].legend(prop={'size': 20})
        a[0].tick_params(axis='x', labelsize=15)
        a[0].tick_params(axis='y', labelsize=15)
        a[1].tick_params(axis='x', labelsize=15)
        a[1].tick_params(axis='y', labelsize=15)
        a[0].set_ylabel('Error / m', size='15')
        # a[0].set(ylim=(0, 0.175))
        # a[1].set(ylim=(0, 0.175))

    _, axs = plt.subplots(2, 2)
    axs[0, 0].plot(poses[:, 1, 0], label='RH, meas x')
    axs[0, 0].plot(upsampled[:, 1, 0], label='est x')
    axs[0, 0].plot(poses[:, 1, 2], label='meas y')
    axs[0, 0].plot(-upsampled[:, 1, 1], label='est y')
    axs[0, 0].plot(poses[:, 1, 1], label='meas z')
    axs[0, 0].plot(upsampled[:, 1, 2], label='est z')
    axs[0, 0].legend()
    axs[0, 1].plot(poses[:, 4, 0], label='LH, meas x')
    axs[0, 1].plot(upsampled[:, 4, 0], label='est x')
    axs[0, 1].plot(poses[:, 4, 2], label='meas y')
    axs[0, 1].plot(-upsampled[:, 4, 1], label='est y')
    axs[0, 1].plot(poses[:, 4, 1], label='meas z')
    axs[0, 1].plot(upsampled[:, 4, 2], label='est z')
    axs[0, 1].legend()
    axs[1, 0].plot(poses[:, 2, 0], label='RK, meas x')
    axs[1, 0].plot(upsampled[:, 2, 0], label='est x')
    axs[1, 0].plot(poses[:, 2, 2], label='meas y')
    axs[1, 0].plot(-upsampled[:, 2, 1], label='est y')
    axs[1, 0].plot(poses[:, 2, 1], label='meas z')
    axs[1, 0].plot(upsampled[:, 2, 2], label='est z')
    axs[1, 0].legend()
    axs[1, 1].plot(poses[:, 3, 0], label='RA, meas x')
    axs[1, 1].plot(upsampled[:, 3, 0], label='est x')
    axs[1, 1].plot(poses[:, 3, 2], label='meas y')
    axs[1, 1].plot(-upsampled[:, 3, 1], label='est y')
    axs[1, 1].plot(poses[:, 3, 1], label='meas z')
    axs[1, 1].plot(upsampled[:, 3, 2], label='est z')
    axs[1, 1].legend()

    _, axs = plt.subplots(2)
    axs[0].plot(-poses[:, 1, 0], label='X')
    axs[0].plot(-poses[:, 1, 2], label='Y')
    axs[0].plot(poses[:, 1, 1], label='Z')
    axs[0].legend()

    axs[1].plot(preds[:, 4, 0], label='X')
    axs[1].plot(preds[:, 4, 1], label='Y')
    axs[1].plot(preds[:, 4, 2], label='Z')
    axs[1].legend()

    _, axs = plt.subplots(2)
    axs[0].plot(-poses[:, 1, 0], label='X')
    axs[0].plot(-poses[:, 1, 2], label='Y')
    axs[0].plot(poses[:, 1, 1], label='Z')
    axs[0].legend()

    axs[1].plot(rotated[:, 4, 0], label='X')
    axs[1].plot(rotated[:, 4, 1], label='Y')
    axs[1].plot(rotated[:, 4, 2], label='Z')
    axs[1].legend()
    #
    # _, axs = plt.subplots(2)
    # axs[0].plot(poses[:, 2, 0], label='X')
    # axs[0].plot(poses[:, 2, 1], label='Y')
    # axs[0].plot(poses[:, 2, 2], label='Z')
    # axs[0].legend()
    #
    # axs[1].plot(preds[:, 2, 0], label='X')
    # axs[1].plot(preds[:, 2, 1], label='Y')
    # axs[1].plot(preds[:, 2, 2], label='Z')
    # axs[1].legend()
    #
    # _, axs = plt.subplots(2)
    # axs[0].plot(poses[:, 3, 0], label='X')
    # axs[0].plot(poses[:, 3, 1], label='Y')
    # axs[0].plot(poses[:, 3, 2], label='Z')
    # axs[0].legend()
    #
    # axs[1].plot(preds[:, 6, 0], label='X')
    # axs[1].plot(preds[:, 6, 1], label='Y')
    # axs[1].plot(preds[:, 6, 2], label='Z')
    # axs[1].legend()

    # fig = plt.figure()
    # vid_frame = 1000
    # # rotated = rotation.apply(preds[vid_frame, ...])
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(preds[vid_frame, :, 0], preds[vid_frame, :, 1],
    #            preds[vid_frame, :, 2], 'b')
    # ax.plot(preds[vid_frame, [1, 0, 4], 0], preds[vid_frame, [1, 0, 4], 1],
    #         preds[10, [1, 0, 4], 2], 'b')
    # ax.plot(preds[vid_frame, [1, 2, 3], 0], preds[vid_frame, [1, 2, 3], 1],
    #         preds[vid_frame, [1, 2, 3], 2], 'b')
    # ax.plot(preds[vid_frame, [4, 5, 6], 0], preds[vid_frame, [4, 5, 6], 1],
    #         preds[vid_frame, [4, 5, 6], 2], 'b')
    # ax.plot(preds[vid_frame, [0, 7, 8, 9, 10], 0],
    #         preds[vid_frame, [0, 7, 8, 9, 10], 1],
    #         preds[vid_frame, [0, 7, 8, 9, 10], 2], 'b')
    # ax.plot(preds[vid_frame, [16, 15, 14, 8, 11, 12, 13], 0],
    #         preds[vid_frame, [16, 15, 14, 8, 11, 12, 13], 1],
    #         preds[vid_frame, [16, 15, 14, 8, 11, 12, 13], 2], 'b')
    # # ax.plot(preds[0, :, 0], preds[0, :, 1], preds[0, :, 2])
    # ax.scatter(poses[int(2.5 * vid_frame), :, 0],
    #            -poses[int(2.5 * vid_frame), :, 2],
    #            poses[int(2.5 * vid_frame), :, 1], 'r')
    # ax.plot(poses[int(2.5 * vid_frame), [1, 0, 4], 0],
    #         -poses[int(2.5 * vid_frame), [1, 0, 4], 2],
    #         poses[int(2.5 * vid_frame), [1, 0, 4], 1], 'r')
    # ax.plot(poses[int(2.5 * vid_frame), [1, 2, 3], 0],
    #         -poses[int(2.5 * vid_frame), [1, 2, 3], 2],
    #         poses[int(2.5 * vid_frame), [1, 2, 3], 1], 'r')
    # ax.plot(poses[int(2.5 * vid_frame), [0, 6, 5], 0],
    #         -poses[int(2.5 * vid_frame), [0, 6, 5], 2],
    #         poses[int(2.5 * vid_frame), [0, 6, 5], 1], 'r')

    # fig = plt.figure()
    # vid_frame = 1000
    # ax = fig.add_subplot(111, projection='3d')
    #
    # fig = plt.figure()
    vid_frame = 100
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(rotated[vid_frame, :, 0], rotated[vid_frame, :, 1],
    #            rotated[vid_frame, :, 2], 'b')
    # ax.plot(rotated[vid_frame, [1, 0, 4], 0], rotated[vid_frame, [1, 0, 4], 1],
    #         rotated[vid_frame, [1, 0, 4], 2], 'b')
    # ax.plot(rotated[vid_frame, [1, 2, 3], 0], rotated[vid_frame, [1, 2, 3], 1],
    #         rotated[vid_frame, [1, 2, 3], 2], 'b')
    # ax.plot(rotated[vid_frame, [4, 5, 6], 0], rotated[vid_frame, [4, 5, 6], 1],
    #         rotated[vid_frame, [4, 5, 6], 2], 'b')
    # ax.plot(rotated[vid_frame, [0, 7, 8, 9, 10], 0],
    #         rotated[vid_frame, [0, 7, 8, 9, 10], 1],
    #         rotated[vid_frame, [0, 7, 8, 9, 10], 2], 'b')
    # ax.plot(rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 0],
    #         rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 1],
    #         rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 2], 'b')
    #
    # # ax.plot(preds[0, :, 0], preds[0, :, 1], preds[0, :, 2])
    # ax.scatter(poses[int(2.5 * vid_frame), :, 0],
    #            -poses[int(2.5 * vid_frame), :, 2],
    #            poses[int(2.5 * vid_frame), :, 1], 'r')
    # ax.plot(poses[int(2.5 * vid_frame), [1, 0, 4], 0],
    #         -poses[int(2.5 * vid_frame), [1, 0, 4], 2],
    #         poses[int(2.5 * vid_frame), [1, 0, 4], 1], 'r')
    # ax.plot(poses[int(2.5 * vid_frame), [1, 2, 3], 0],
    #         -poses[int(2.5 * vid_frame), [1, 2, 3], 2],
    #         poses[int(2.5 * vid_frame), [1, 2, 3], 1], 'r')
    # ax.plot(poses[int(2.5 * vid_frame), [0, 6, 5], 0],
    #         -poses[int(2.5 * vid_frame), [0, 6, 5], 2],
    #         poses[int(2.5 * vid_frame), [0, 6, 5], 1], 'r')

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    # ax.set_aspect('equal')

    ax.scatter(rotated[vid_frame, :, 0], rotated[vid_frame, :, 1],
               rotated[vid_frame, :, 2], 'b')
    ax.plot(rotated[vid_frame, [1, 0, 4], 0], rotated[vid_frame, [1, 0, 4], 1],
            rotated[vid_frame, [1, 0, 4], 2], 'b')
    ax.plot(rotated[vid_frame, [1, 2, 3], 0], rotated[vid_frame, [1, 2, 3], 1],
            rotated[vid_frame, [1, 2, 3], 2], 'b')
    ax.plot(rotated[vid_frame, [4, 5, 6], 0], rotated[vid_frame, [4, 5, 6], 1],
            rotated[vid_frame, [4, 5, 6], 2], 'b')
    ax.plot(rotated[vid_frame, [0, 7, 8, 9, 10], 0],
            rotated[vid_frame, [0, 7, 8, 9, 10], 1],
            rotated[vid_frame, [0, 7, 8, 9, 10], 2], 'b')
    ax.plot(rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 0],
            rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 1],
            rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 2], 'b')

    ax.scatter(poses[int(2.5 * vid_frame), :, 0],
               -poses[int(2.5 * vid_frame), :, 2],
               poses[int(2.5 * vid_frame), :, 1], 'r')
    ax.plot(poses[int(2.5 * vid_frame), [1, 0, 4], 0],
            -poses[int(2.5 * vid_frame), [1, 0, 4], 2],
            poses[int(2.5 * vid_frame), [1, 0, 4], 1], 'r')
    ax.plot(poses[int(2.5 * vid_frame), [1, 2, 3], 0],
            -poses[int(2.5 * vid_frame), [1, 2, 3], 2],
            poses[int(2.5 * vid_frame), [1, 2, 3], 1], 'r')
    ax.plot(poses[int(2.5 * vid_frame), [0, 6, 5], 0],
            -poses[int(2.5 * vid_frame), [0, 6, 5], 2],
            poses[int(2.5 * vid_frame), [0, 6, 5], 1], 'r')

    # ax.autoscale_view(scalex=False, scaley=False, scalez=False)
    axisEqual3D(ax)

    ax = fig.add_subplot(122, projection='3d')
    # ax.set_aspect('equal')
    vid_frame = 1000
    ax.scatter(rotated[vid_frame, :, 0], rotated[vid_frame, :, 1],
               rotated[vid_frame, :, 2], 'b')
    ax.plot(rotated[vid_frame, [1, 0, 4], 0], rotated[vid_frame, [1, 0, 4], 1],
            rotated[vid_frame, [1, 0, 4], 2], 'b', label='Estimated joints')
    ax.plot(rotated[vid_frame, [1, 2, 3], 0], rotated[vid_frame, [1, 2, 3], 1],
            rotated[vid_frame, [1, 2, 3], 2], 'b')
    ax.plot(rotated[vid_frame, [4, 5, 6], 0], rotated[vid_frame, [4, 5, 6], 1],
            rotated[vid_frame, [4, 5, 6], 2], 'b')
    ax.plot(rotated[vid_frame, [0, 7, 8, 9, 10], 0],
            rotated[vid_frame, [0, 7, 8, 9, 10], 1],
            rotated[vid_frame, [0, 7, 8, 9, 10], 2], 'b')
    ax.plot(rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 0],
            rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 1],
            rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 2], 'b')

    ax.scatter(poses[int(2.5 * vid_frame), :, 0],
               -poses[int(2.5 * vid_frame), :, 2],
               poses[int(2.5 * vid_frame), :, 1], 'r')
    ax.plot(poses[int(2.5 * vid_frame), [1, 0, 4], 0],
            -poses[int(2.5 * vid_frame), [1, 0, 4], 2],
            poses[int(2.5 * vid_frame), [1, 0, 4], 1], 'r', label='Measured joints')
    ax.plot(poses[int(2.5 * vid_frame), [1, 2, 3], 0],
            -poses[int(2.5 * vid_frame), [1, 2, 3], 2],
            poses[int(2.5 * vid_frame), [1, 2, 3], 1], 'r')
    ax.plot(poses[int(2.5 * vid_frame), [0, 6, 5], 0],
            -poses[int(2.5 * vid_frame), [0, 6, 5], 2],
            poses[int(2.5 * vid_frame), [0, 6, 5], 1], 'r')

    # ax.autoscale_view(scalex=False, scaley=False, scalez=False)
    axisEqual3D(ax)
    ax.legend(loc=2, prop={'size': 20})
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='y', labelsize=20)
    # print(preds[5, :, :].shape)
    # ax.scatter(preds[5, :, :])
    # ax.scatter(poses[25, :, :])
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.set_zlabel('Z / m')
    # ax.axis('equal')
    # numpoints = 10
    # points = np.random.random((2, numpoints))
    # colors = cm.rainbow(np.linspace(0, 1, numpoints))

    plt.show()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                        for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == '__main__':
    main()
