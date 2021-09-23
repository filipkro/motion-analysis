import sys
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


sys.path.append(os.path.join(BASE, 'pose/analysis'))
sys.path.append(os.path.join(BASE, 'pose/analysis/utils'))
sys.path.append(os.path.join(BASE, 'classification/tsc/utils'))

# for p in sys.path:
#     print(p)
# # print()
# print(os.path.dirname(os.path.realpath(__file__)))
# print(BASE)

# from analyse_vid import start as start_detection

# from analyse_vid_light import start as start_detection

# from eval_vid import main as assess_subject
from argparse import ArgumentParser, ArgumentTypeError, Namespace

def run_video_detection(args):
    from analyse_vid_light import start as start_detection

    # base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    vid_args = Namespace()
    # vid_args.video_path = args.video
    vid_args.video_path = os.path.join(BASE, '03SLS1R_MUSSE.mts')
    # vid_args.pose_config = os.path.join(BASE, 'pose/mmpose-files/hrnet_w48_coco_wholebody_384x288_dark.py')
    # vid_args.pose_checkpoint = os.path.join(BASE, 'pose/mmpose-files/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')
    vid_args.pose_config = os.path.join(BASE, 'pose/mmpose-files/hrnet_w32_coco_wholebody_256x192_dark.py')
    vid_args.pose_checkpoint = os.path.join(BASE, 'pose/mmpose-files/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth')
    vid_args.folder_box = os.path.join(BASE, 'pose/mmdet-files')
    vid_args.show = False
    vid_args.out_video_root = args.out
    vid_args.device = 'cpu'
    vid_args.box_thr = 0.1
    vid_args.kpt_thr = 0.1
    vid_args.file_name = ''
    vid_args.show_box = False
    vid_args.allow_flip = False
    vid_args.save_pixels = False
    vid_args.save4_3d = False
    vid_args.flip2right = True
    vid_args.fname_format = True
    vid_args.skip_rate = True
    vid_args.only_box = False
    vid_args.save_numpy = False

    poses, meta, fp = start_detection(vid_args)
    print(fp)
    return poses, meta['fps']


def extract_reps(data, fps):
    from extract_reps import main as get_datasets

    rep_args = Namespace()
    rep_args.filepath = ''
    rep_args.debug = False
    rep_args.rate = 25
    rep_args.save_numpy = False

    return get_datasets(rep_args, data, fps)


def main(args):

    poses, fps = run_video_detection(args)
    datasets, datasets100 = extract_reps(poses, fps)

    from eval_vid import main as assess_subject
    assess_subject(Namespace(), datasets=datasets, datasets100=datasets100, base_path=BASE)


if __name__ == '__main__':
    print('starting...')
    parser = ArgumentParser()
    # parser.add_argument('video', help='Video to analyse')
    parser.add_argument('--out', help='Directory in which results are saved, Default is ./out', default='out')
    args = parser.parse_args()
    main(args)
