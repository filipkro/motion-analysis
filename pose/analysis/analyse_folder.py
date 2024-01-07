from argparse import ArgumentParser, ArgumentTypeError
import os
import analyse_vid
import numpy as np
import utils.dict_utils as du

import torch

FILE_FORMATS = ('.avi', '.mp4', '.MTS', '.MOV', '.mp2t', '.mts')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def main():

    print('CUDA enabled:', torch.cuda.is_available())

    assert torch.cuda.is_available()

    print('parsing arguments...')
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('video_folder', type=str, help='Video folder')
    parser.add_argument('--show', type=str2bool, nargs='?', const=True,
                        default=False, help="show results.")
    parser.add_argument('--out-video-root', default='',
                        help='Root of the output video file. '
                        'Default not saving the visualization video.')
    parser.add_argument('--device', default='cpu',
                        help='Device used for inference')
    parser.add_argument('--box-thr', type=float, default=0.1,
                        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.1,
                        help='Keypoint score threshold')
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--only_box', type=str2bool, nargs='?', const=True,
                        default=False, help="only show bounding box")
    parser.add_argument('--folder_box', type=str, default='')
    parser.add_argument('--show_box', type=str2bool, nargs='?', const=True,
                        default=False, help="show bounding box.")
    parser.add_argument('--allow_flip', type=str2bool, nargs='?', const=True,
                        default=True, help='for FL')
    parser.add_argument('--save_pixels', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='saveposes as pixels or ratio of im')
    parser.add_argument('--save4_3d', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='save poses in format for VidePose3D.')
    parser.add_argument('--flip2right', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='flips video if name contains L')
    parser.add_argument('--fname_format', type=str2bool, nargs='?',
                        default=True,
                        help='if filename has format of marked videos')
    parser.add_argument('--skip_rate', type=int, default=1)
    parser.add_argument('--keep_fname', type=str2bool, nargs='?',
                        default=True,
                        help='assumes new format of fname, which will be kept - fix this')
    args = parser.parse_args()

    if not args.fname_format:
        args.flip2right = False

    if args.flip2right:
        args.allow_flip = False
    print('arguments parsed, starting')

    # i have only access to gpu on cluster, hence:
    if args.device != 'cpu':
        args.show = False
        args.only_box = False
        args.show_box = False

    if args.video_folder[-1] != '/':
        args.video_folder += '/'

    print(args.video_folder)

    print('files to be processed: {0}'.format(os.listdir(args.video_folder)))

    nbr_of_files = 0
    for _, _, files in os.walk(args.video_folder):
        for vid in files:
            if vid.endswith(FILE_FORMATS):
                nbr_of_files += 1
    processed = 1

    meta_data = {}
    meta_data = {'layout_name': 'coco', 'num_joints': 17,
                 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],
                                        [2, 4, 6, 8, 10, 12, 14, 16]],
                 'video_metadata': {}}

    poses_data = {}

    # for root, directories, files in os.walk(args.video_folder):
    #     print(directories)
    # for files in os.listdir(args.video_folder):
    #     for vid in os.listdir(args.video_folder + files):
    for vid in os.listdir(args.video_folder):
        print(vid)
        if vid.endswith(FILE_FORMATS):
            args.video_path = args.video_folder + vid
            print(vid)
            poses, meta, name = analyse_vid.start(args)

            print('Video {:.0f} out of {:.0f}'.format(
                processed, nbr_of_files))

            poses_data = du.generate_dict(name, poses, '2d', poses_data,
                                          meta)
            meta_data['video_metadata'][name.split('_')[0]] = meta

        processed += 1

    save_name = args.out_video_root + '/data_2d_custom_' + 'complete.npz'
    print(save_name)
    np.savez_compressed(save_name, positions_2d=poses_data, metadata=meta_data)
    print('DONE')


if __name__ == '__main__':
    print('starting...')
    main()
