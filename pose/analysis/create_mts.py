import numpy as np
# import filt
from argparse import ArgumentParser
from split_sequence import split
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def parse_mts(poses, kpts, label, mts=None, mts_labels=None, angle_pts=None,
              debug=False):

    print('')
    print('line 11 {0}'.format(poses.shape))
    split_poses, _ = split(poses, debug=debug)
    print('split', split_poses.shape)

    if angle_pts is not None:
        ang = calc_angle(split_poses, angle_pts)

    print('kpots', kpts)
    kpts.append(13)
    print('kpots', kpts)
    split_poses = split_poses[:, :, kpts, :]

    sh = split_poses.shape
    print('line 23 {0}'.format(sh))
    split_poses = split_poses.reshape((sh[0], sh[1], sh[2] * sh[3]))

    if angle_pts is not None:
        xtra = np.expand_dims(split_poses[:, :, -1], axis=-1)
        split_poses = np.append(split_poses[:, :, :-1], ang, axis=-1)
        split_poses = np.append(split_poses, xtra, axis=-1)

    print('features: {0}'.format(split_poses.shape[2]))
    if mts is not None:
        print('mts features: {0}'.format(mts.shape[2]))
        assert mts.shape[2] == split_poses.shape[2]

    if mts is not None:
        # print('mts', mts.shape)
        min_len = np.min((mts.shape[1], split_poses.shape[1]))
        # print(min_len)
        if min_len < split_poses.shape[1]:
            # print(peaks)
            # crop = np.zeros(mts.shape)
            crop = np.zeros((split_poses.shape[0], min_len,
                             split_poses.shape[2]))
            p_len = split_poses.shape[1]
            for i in range(split_poses.shape[0]):
                peak, _ = find_peaks(split_poses[i, :, -1], distance=50,
                                     prominence=0.05, width=13)
                print(peak)
                print(i)
                # plt.plot(split_poses[i, :, -1])
                # plt.show()
                # if len(peak) > 0:
                print(len(peak))
                peak = peak[0]

                if peak <= int(min_len / 2):
                    crop[i, ...] = split_poses[i, :min_len, ...]
                elif peak <= (p_len - int(min_len / 2)):
                    print('IN IF')
                    print(int(np.ceil(min_len / 2)))
                    print(len(split_poses[i, peak - int(np.ceil(min_len / 2)):
                                          peak - int(np.ceil(min_len / 2))
                                          + min_len, ...]))
                    crop[i, ...] = split_poses[i, peak -
                                               int(np.ceil(min_len / 2)):
                                               peak -
                                               int(np.ceil(min_len / 2))
                                               + min_len, ...]
                else:
                    crop[i, ...] = split_poses[i, -min_len:, ...]

                # if ((peak >= int(min_len / 2)) and
                #         (peak <= p_len - int(min_len / 2))):
                #     crop[i, ...] = split_poses[i, peak - int(min_len / 2):
                #                                peak +
                #                                int(np.ceil(min_len / 2)), ...]
                # elif peak < p_len / 2:
                #     crop[i, ...] = split_poses[i, :min_len, ...]
                # else:
                #     crop[i, ...] = split_poses[i, -min_len:, ...]

            split_poses = crop

        elif min_len < mts.shape[1]:
            # crop = np.zeros(split_poses.shape)
            crop = np.zeros((mts.shape[0], min_len, mts.shape[2]))
            p_len = mts.shape[1]
            for i in range(mts.shape[0]):
                peak, _ = find_peaks(mts[i, :, -1], distance=50,
                                     prominence=0.05, width=13)
                print(peak)
                if len(peak) == 0:
                    plt.plot(mts[i, :, -1])
                    plt.show()
                peak = peak[0]
                if ((peak >= int(min_len / 2)) and
                        (peak <= p_len - int(min_len / 2))):

                    print(i)
                    print(peak - int(min_len / 2))
                    print(peak + int(np.ceil(min_len / 2)))
                    print(crop.shape)
                    print(mts.shape)
                    crop[i, ...] = mts[i, peak - int(min_len / 2):
                                       peak + int(np.ceil(min_len / 2)), ...]
                elif peak < p_len / 2:
                    crop[i, ...] = mts[i, :min_len, ...]
                else:
                    crop[i, ...] = mts[i, -min_len:, ...]

            mts = crop

        mts = np.append(mts, split_poses, axis=0)
        mts_labels = np.append(mts_labels,
                               label * np.ones(split_poses.shape[0]))

    else:
        mts = split_poses
        mts_labels = label * np.ones(split_poses.shape[0])

    print('Shape of MTS: {0}', format(mts.shape))
    # plt.plot(mts[:, :, -1].T)
    # plt.show()

    return mts, mts_labels


def calc_angle(poses, kpts):
    angles = None
    for pt in kpts:
        ang = np.expand_dims(np.arctan2(poses[..., pt[0], 1] -
                                        poses[..., pt[1], 1],
                                        poses[..., pt[0], 0] -
                                        poses[..., pt[1], 0]), axis=-1)

        angles = ang if angles is None else np.append(angles, ang, axis=-1)

    print('dims in calac, {0}, {1}'.format(poses.shape, angles.shape))

    return angles


def fake_mts(file, kpts, angles=None):
    mts = np.load(file)
    mts = mts[6:-1, :, 0:2]
    mts, _ = split(mts)
    if angles is not None:
        ang = calc_angle(mts, angles)
    kpts.append(14)
    mts = mts[:, :, kpts, :]
    sh = mts.shape
    mts = mts.reshape((sh[0], sh[1], sh[2] * sh[3]))

    if angles is not None:
        xtra = mts[:, :, -1]
        mts = np.append(mts[:, :, :-1], ang, axis=-1)
        mts = np.append(mts, np.expand_dims(xtra, axis=-1), axis=-1)
        # mts = np.insert(mts, -1, ang, axis=2)

    # plt.plot(mts[..., -1].T)
    # plt.show()

    return mts


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    parser.add_argument('--np_file2', default='', help='Numpy file to filter')
    args = parser.parse_args()

    poses = np.load(args.np_file)

    poses = poses[6: -1, :, 0: 2]
    kpts = [12, 13, 15]
    if args.np_file2 != '':
        # ang = calc_angle(poses, 11, 13)
        angles = np.array([[11, 13], [13, 15]])
        mts = fake_mts(args.np_file2, kpts.copy(), angles=angles)
        parse_mts(poses, kpts, 1, mts=mts,
                  mts_labels=np.ones(mts.shape[1]), angle_pts=angles)
    else:
        parse_mts(poses, kpts, 1)


if __name__ == '__main__':
    main()
