import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from celluloid import Camera
from matplotlib import cm
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
    center_pred = (preds[:, 12, :] + preds[:, 11, :]) / 2
    preds = np.insert(preds, -1, center_pred, axis=1)
    # preds = np.insert(preds, 0, center_pred, axis=1)
    preds = preds - np.expand_dims(preds[:, -2, :], 1)
    print('sum', np.sum(preds[:, -1, :]))
    dpreds = np.sqrt((np.mean(preds[:, 11, 0]) - np.mean(preds[:, 12, 0]))**2 +
                     (np.mean(preds[:, 11, 1]) - np.mean(preds[:, 12, 1]))**2)

    poses = extract_joints(truth)
    dtruths = np.sqrt((np.mean(poses[:, 1, 0]) - np.mean(poses[:, 4, 0]))**2 +
                      (np.mean(poses[:, 1, 2]) - np.mean(poses[:, 4, 2]))**2)
    preds = preds * dtruths / dpreds
    # truth = truth[]

    rotation_radians = -np.pi / 9
    rotation_axis = np.array([1, 0, 0])
    rotation_vector = rotation_radians * rotation_axis
    rotation = Rot.from_rotvec(rotation_vector)
    print(preds.shape)
    rotated = np.zeros((preds.shape[0], preds.shape[1], 3))
    for i in range(preds.shape[0]):
        rotated[i, ...] = rotation.apply(
            np.append(preds[i, ...], np.zeros((preds.shape[1], 1)), axis=1))

    print(np.sum(rotated[:, :, :-1] - preds))

    ratio = preds.shape[0] / poses.shape[0]
    upsampled = resample(preds, ratio)
    print(upsampled.shape)
    print(poses.shape)
    error_rh = np.sqrt((poses[:, 1, 0] - upsampled[:, 12, 0])
                       ** 2 + (poses[:, 1, 2] + upsampled[:, 12, 1])**2)
    error_lh = np.sqrt((poses[:, 4, 0] - upsampled[:, 11, 0])
                       ** 2 + (poses[:, 4, 2] + upsampled[:, 11, 1])**2)
    error_rk = np.sqrt((poses[:, 2, 0] - upsampled[:, 14, 0])
                       ** 2 + (poses[:, 2, 2] + upsampled[:, 14, 1])**2)
    error_ra = np.sqrt((poses[:, 3, 0] - upsampled[:, 17, 0])
                       ** 2 + (poses[:, 3, 2] + upsampled[:, 17, 1])**2)

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
        a[0].set(ylim=(0, 0.175))
        a[1].set(ylim=(0, 0.175))
        # a[1].set_ylabel('Error / m', size='15')

    _, axs = plt.subplots(2, 2)
    axs[0, 0].plot(poses[:, 1, 0], label='meas x')
    axs[0, 0].plot(upsampled[:, 12, 0], label='est x')
    axs[0, 0].plot(poses[:, 1, 2], label='meas y')
    axs[0, 0].plot(-upsampled[:, 12, 1], label='est y')
    axs[0, 0].legend()
    axs[0, 1].plot(poses[:, 4, 0], label='meas x')
    axs[0, 1].plot(upsampled[:, 11, 0], label='est x')
    axs[0, 1].plot(poses[:, 4, 2], label='meas y')
    axs[0, 1].plot(-upsampled[:, 11, 1], label='est y')
    axs[0, 1].legend()
    axs[1, 0].plot(poses[:, 2, 0], label='meas x')
    axs[1, 0].plot(upsampled[:, 14, 0], label='est x')
    axs[1, 0].plot(poses[:, 2, 2], label='meas y')
    axs[1, 0].plot(-upsampled[:, 14, 1], label='est y')
    axs[1, 0].legend()
    axs[1, 1].plot(poses[:, 3, 0], label='meas x')
    axs[1, 1].plot(upsampled[:, 17, 0], label='est x')
    axs[1, 1].plot(poses[:, 3, 2], label='meas y')
    axs[1, 1].plot(-upsampled[:, 17, 1], label='est y')
    axs[1, 1].legend()
    # _, axs = plt.subplots(2)
    # axs[0].plot(-poses[:, 1, 0], label='X')
    # # axs[0].plot(-poses[:, 1, 2], label='Y')
    # axs[0].plot(poses[:, 1, 1], label='Z')
    # axs[0].legend()
    #
    # axs[1].plot(preds[:, 12, 0], label='X')
    # axs[1].plot(preds[:, 12, 1], label='Y')
    # # axs[1].plot(preds[:, 4, 2], label='Z')
    # axs[1].legend()
    #
    # _, axs = plt.subplots(2)
    # axs[0].plot(-poses[:, 1, 0], label='X')
    # # axs[0].plot(-poses[:, 1, 2], label='Y')
    # axs[0].plot(poses[:, 1, 1], label='Z')
    # axs[0].legend()
    #
    # axs[1].plot(rotated[:, 4, 0], label='X')
    # axs[1].plot(rotated[:, 4, 1], label='Y')
    # # axs[1].plot(rotated[:, 4, 2], label='Z')
    # axs[1].legend()
    # #
    # _, axs = plt.subplots(2)
    # axs[0].plot(poses[:, 2, 0], label='X')
    # # axs[0].plot(poses[:, 2, 1], label='Y')
    # axs[0].plot(poses[:, 2, 2], label='Z')
    # axs[0].legend()
    #
    # axs[1].plot(preds[:, 2, 0], label='X')
    # axs[1].plot(preds[:, 2, 1], label='Y')
    # # axs[1].plot(preds[:, 2, 2], label='Z')
    # axs[1].legend()
    # #
    # _, axs = plt.subplots(2)
    # axs[0].plot(poses[:, 3, 0], label='X')
    # # axs[0].plot(poses[:, 3, 1], label='Y')
    # axs[0].plot(poses[:, 3, 2], label='Z')
    # axs[0].legend()
    #
    # axs[1].plot(preds[:, 6, 0], label='X')
    # axs[1].plot(preds[:, 6, 1], label='Y')
    # # axs[1].plot(preds[:, 6, 2], label='Z')
    # axs[1].legend()

    # fig = plt.figure()
    fig, axs = plt.subplots(1, 2)
    vid_frame = 100
    vid_frame2 = 1000
    axs[0].scatter(-preds[vid_frame, :, 0], -preds[vid_frame, :, 1], c='b')
    axs[0].plot(-preds[vid_frame, [17, 14, 12, 16, 11, 13, 15], 0],
                -preds[vid_frame, [17, 14, 12, 16, 11, 13, 15], 1], c='b',
                label='Estimated joints')
    axs[0].plot(-preds[vid_frame, [11, 5, 6, 12], 0],
                -preds[vid_frame, [11, 5, 6, 12], 1], c='b')
    axs[0].plot(-preds[vid_frame, [9, 7, 5, 3, 1, 2, 4, 6, 8, 10], 0],
                -preds[vid_frame, [9, 7, 5, 3, 1, 2, 4, 6, 8, 10], 1], c='b')
    axs[0].plot(-preds[vid_frame, [1, 0, 2], 0],
                -preds[vid_frame, [1, 0, 2], 1], c='b')
    # plt.plot(-)
    # plt.figure()
    axs[0].scatter(poses[int(2.5 * vid_frame), :, 0],
                   poses[int(2.5 * vid_frame), :, 2], c='r')

    axs[0].plot(poses[int(2.5 * vid_frame), [1, 0, 4], 0],
                poses[int(2.5 * vid_frame), [1, 0, 4], 2], c='r',
                label='Measured joints')
    axs[0].plot(poses[int(2.5 * vid_frame), [1, 2, 3], 0],
                poses[int(2.5 * vid_frame), [1, 2, 3], 2], c='r')
    axs[0].plot(poses[int(2.5 * vid_frame), [0, 6, 5], 0],
                poses[int(2.5 * vid_frame), [0, 6, 5], 2], c='r')
    axs[0].plot(poses[int(2.5 * vid_frame), [8, 5, 7], 0],
                poses[int(2.5 * vid_frame), [8, 5, 7], 2], c='r')
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    # axs[0].gca().set_aspect('equal', adjustable='box')

    # axs[0].axis('off')
    axs[0].axis('equal')
    axs[1].scatter(-preds[vid_frame2, :, 0], -preds[vid_frame2, :, 1], c='b')
    axs[1].plot(-preds[vid_frame2, [17, 14, 12, 16, 11, 13, 15], 0],
                -preds[vid_frame2, [17, 14, 12, 16, 11, 13, 15], 1], c='b',
                label='Estimated joints')
    axs[1].plot(-preds[vid_frame2, [11, 5, 6, 12], 0],
                -preds[vid_frame2, [11, 5, 6, 12], 1], c='b')
    axs[1].plot(-preds[vid_frame2, [9, 7, 5, 3, 1, 2, 4, 6, 8, 10], 0],
                -preds[vid_frame2, [9, 7, 5, 3, 1, 2, 4, 6, 8, 10], 1], c='b')
    axs[1].plot(-preds[vid_frame2, [1, 0, 2], 0],
                -preds[vid_frame2, [1, 0, 2], 1], c='b')
    # plt.plot(-)
    # plt.figure()
    axs[1].scatter(poses[int(2.5 * vid_frame2), :, 0],
                   poses[int(2.5 * vid_frame2), :, 2], c='r')

    axs[1].plot(poses[int(2.5 * vid_frame2), [1, 0, 4], 0],
                poses[int(2.5 * vid_frame2), [1, 0, 4], 2], c='r',
                label='Measured joints')
    axs[1].plot(poses[int(2.5 * vid_frame2), [1, 2, 3], 0],
                poses[int(2.5 * vid_frame2), [1, 2, 3], 2], c='r')
    axs[1].plot(poses[int(2.5 * vid_frame2), [0, 6, 5], 0],
                poses[int(2.5 * vid_frame2), [0, 6, 5], 2], c='r')
    axs[1].plot(poses[int(2.5 * vid_frame2), [8, 5, 7], 0],
                poses[int(2.5 * vid_frame2), [8, 5, 7], 2], c='r')
    axs[1].axis('equal')
    axs[1].legend(loc='best', prop={'size': 20})
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    # axs[1].axis('off')
    # rotated = rotation.apply(preds[vid_frame, ...])
    # plt.figure()
    # plt.scatter(-rotated[vid_frame, :, 0], -rotated[vid_frame, :, 1], c='b')
    # plt.plot(-rotated[vid_frame, [17, 14, 12, 16, 11, 13, 15], 0],
    #          -rotated[vid_frame, [17, 14, 12, 16, 11, 13, 15], 1], c='b')
    # # plt.plot(-)
    # # plt.figure()
    # plt.scatter(poses[int(2.5 * vid_frame), :, 0],
    #             poses[int(2.5 * vid_frame), :, 2], c='r')
    #
    # plt.plot(poses[int(2.5 * vid_frame), [1, 0, 4], 0],
    #          poses[int(2.5 * vid_frame), [1, 0, 4], 2], c='r')
    # plt.plot(poses[int(2.5 * vid_frame), [1, 2, 3], 0],
    #          poses[int(2.5 * vid_frame), [1, 2, 3], 2], c='r')
    # plt.plot(poses[int(2.5 * vid_frame), [0, 6, 5], 0],
    #          poses[int(2.5 * vid_frame), [0, 6, 5], 2], c='r')

    # fig = plt.figure()
    # vid_frame = 1000
    # ax = fig.add_subplot(111, projection='3d')

    # fig = plt.figure()
    # vid_frame = 1000
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(rotated[vid_frame, :, 0], rotated[vid_frame, :, 1],
    #            rotated[vid_frame, :, 2], color='b')
    # ax.plot(rotated[vid_frame, [1, 0, 4], 0], rotated[vid_frame, [1, 0, 4], 1],
    #         rotated[vid_frame, [1, 0, 4], 2], color='b')
    # ax.plot(rotated[vid_frame, [1, 2, 3], 0], rotated[vid_frame, [1, 2, 3], 1],
    #         rotated[vid_frame, [1, 2, 3], 2], color='b')
    # ax.plot(rotated[vid_frame, [4, 5, 6], 0], rotated[vid_frame, [4, 5, 6], 1],
    #         rotated[vid_frame, [4, 5, 6], 2], color='b')
    # ax.plot(rotated[vid_frame, [0, 7, 8, 9, 10], 0],
    #         rotated[vid_frame, [0, 7, 8, 9, 10], 1],
    #         rotated[vid_frame, [0, 7, 8, 9, 10], 2], color='b')
    # ax.plot(rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 0],
    #         rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 1],
    #         rotated[vid_frame, [16, 15, 14, 8, 11, 12, 13], 2], color='b')

    # ax.plot(preds[0, :, 0], preds[0, :, 1], preds[0, :, 2])
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

    # print(preds[5, :, :].shape)
    # ax.scatter(preds[5, :, :])
    # ax.scatter(poses[25, :, :])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    plt.show()

    # camera = Camera(plt.figure())
    # a = 0
    # for vid_frame in rotated:
    #     print(a)
    #     plt.scatter(vid_frame[:, 0], -vid_frame[:, 1])
    #     plt.plot(vid_frame[[17, 14, 12, 16, 11, 13, 15], 0], -
    #              vid_frame[[17, 14, 12, 16, 11, 13, 15], 1])
    #     camera.snap()
    #     a += 1
    # anim = camera.animate(blit=True)
    # anim.save('/home/filipkr/Desktop/scatter.mp4')


if __name__ == '__main__':
    main()
