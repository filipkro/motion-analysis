import numpy as np
from argparse import ArgumentParser


def generate_dict(name, poses, dim, data={}, meta=None):
    dim_key = 'positions_' + dim
    subject = name[0:2]
    action = name.split('_')[0][2:]

    if subject not in data:
        data[subject] = {}

    if action not in data[subject]:
        data[subject][action] = {}

    data[subject][action][dim_key] = [poses.astype('float32')]
    if meta is not None:
        data[subject][action]['metadata'] = meta

    return data


def merge(args):
    poses_3d = np.load(args.path_3d, allow_pickle=True)
    poses_3d = poses_3d['poses_3d'].item()
    poses_3d = poses_3d[args.name]
    poses_3d = np.array(poses_3d)

    data = generate_dict(args.name, poses_3d, '3d')
    poses_2d = np.load(args.path_2d)
    width = 960
    height = 544
    meta = {}
    meta['video_metadata'] = {}
    meta['video_metadata'][args.name] = {}
    meta = {'layout_name': 'coco', 'num_joints': 17,
            'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],
                                   [2, 4, 6, 8, 10, 12, 14, 16]],
            'video_metadata': {args.name: {'w': width, 'h': height,
                                           'fps': 60}}}

    data = generate_dict(args.name, poses_2d, '2d', data, meta)

    save_file = '/home/filipkr/Documents/xjob/custom_2d_training.npz'
    np.savez_compressed(save_file, data=data)


def main():
    first = {'01': {'SLS': {'positions_2d': np.array([[1, 2, 3], [4, 5, 6]])}}}
    sec = generate_dict('01FL1R_lol', np.array([[12, 13, 14], [1, 1, 1]]),
                        '3d', first)
    print(sec)
    third = generate_dict('01SLS_lol', np.array([[1, 2], [3, 4], [5, 6]]),
                          '3d', sec)
    print(third)
    fourth = generate_dict('02SLS_cool', np.array([[1], [2]]), '2d', third)
    print(fourth)
    print(generate_dict('01SLS_lol', np.array([[1, 2], [3, 4], [5, 6]]), '3d'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_3d')
    parser.add_argument('--path_2d')
    parser.add_argument('--name')
    args = parser.parse_args()
    merge(args)
    # main()
