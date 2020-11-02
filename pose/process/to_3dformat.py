import numpy as np
from argparse import ArgumentParser
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('poses', help='Config file for pose')
    parser.add_argument('--video_name', type=str, default='lol')
    parser.add_argument('--out_name', type=str, default='lol')

    args = parser.parse_args()
    root = os.path.dirname(args.poses)
    poses = np.load(args.poses)
    poses2 = {}
    poses2[args.video_name] = {}
    poses2[args.video_name]['custom'] = [poses.astype('float32')]
    # poses = {'custom': poses}
    # poses = {args.name: poses}
    # poses = []
    # print(poses2)
    # tt = poses.item()
    # print(tt)
    print(poses2[args.video_name]['custom'][0].shape)
    width = 960
    height = 544
    meta = {}
    meta['video_metadata'] = {}
    meta['video_metadata'][args.video_name] = {}
    meta = {'layout_name': 'coco', 'num_joints': 17,
            'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],
                                   [2, 4, 6, 8, 10, 12, 14, 16]],
            'video_metadata': {args.video_name: {'w': width, 'h': height}}}

    output_prefix_2d = 'data_2d_custom_'

    # file_name = root + '/' + output_prefix_2d + args.out_name
    file_name = '/home/filipkr/Documents/xjob/motion-analysis/pose/' \
        + 'VideoPose3D/data/' + output_prefix_2d + args.out_name
    np.savez_compressed(file_name, positions_2d=poses2, metadata=meta)
    print(meta)
    print(root)


if __name__ == '__main__':
    main()
