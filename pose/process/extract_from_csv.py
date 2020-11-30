import numpy as np
from argparse import ArgumentParser
import pandas as pd
from scipy.interpolate import interp1d

# ORDER: {RIGTH HIP, RIGHT KNEE, RIGHT ANKLE, LEFT HIP, MID_STERN, MID_XIP,
#           RIGHT SHOULDER, LEFT SHOULDER}


def resample(x, factor, kind='linear'):
    n = int(np.ceil(x.shape[0] / factor))
    f = interp1d(np.linspace(0, 1, x.shape[0]), x, kind, axis=0)
    return f(np.linspace(0, 1, n))


def main(args):

    poses_3d = {}
    fields = args.fields.split(',')
    df = pd.read_csv(args.file, delimiter=args.delim)

    for field in fields:
        data = df.filter(like=field).values
        poses = data[4:, -24:].astype(np.float)

        print(poses.shape)
        poses = np.reshape(poses, (poses.shape[0], 8, 3))
        print(poses.shape)
        tmp = poses[:, :, 1].copy()
        poses[:, :, 1] = -poses[:, :, 2]
        poses[:, :, 2] = tmp

        ratio = 150 / 60
        poses = resample(poses, ratio)

        center_hip = (poses[:, 0, :] + poses[:, 3, :]) / 2
        poses = np.insert(poses, 0, center_hip, axis=1)

        print(poses.shape)

        dx = np.max(poses[:, :, 0]) - np.min(poses[:, :, 0])
        # dy = np.max(poses[:, :, 1]) - np.min(poses[:, :, 1])
        dz = np.max(poses[:, :, 2]) - np.min(poses[:, :, 2])
        # norm_dist = np.max((dx, dy, dz))
        norm_dist = np.max((dx, dz))
        poses = 2 * poses / norm_dist - 1
        poses = poses - np.expand_dims(poses[:, 0, :], 1)

        poses_3d[field.split('.')[0]] = poses

    if args.save_path != '':
        np.savez_compressed(args.save_path, poses_3d=poses_3d)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file', help='File')
    parser.add_argument(
        'fields', help='field(s) to extract from csv file, if more than one: '
        + 'separate with ,')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--delim', default='\t')
    args = parser.parse_args()
    main(args)
