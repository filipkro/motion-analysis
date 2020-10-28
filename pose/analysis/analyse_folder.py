from argparse import ArgumentParser, ArgumentTypeError
import os
import analyse_vid


FILE_FORMATS = ('.avi', '.mp4', '.MTS', '.MOV', 'mp2t')


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
    parser.add_argument('--box-thr', type=float, default=0.3,
                        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
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

    args = parser.parse_args()

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

    nbr_of_files = len(os.listdir(args.video_folder))
    processed = 1

    for vid in os.listdir(args.video_folder):
        if vid.endswith(FILE_FORMATS):
            args.video_path = args.video_folder + vid
            print(vid)
            process_time = analyse_vid.start(args)
            t = time.perf_counter()
            print('Video {:.0f} out of {:.0f} processed in {:.4f} \
                seconds.'.format(processed, nbr_of_files, process_time))
        

        processed += 1

    print('DONE')


if __name__ == '__main__':
    print('starting...')
    main()
