import os
from argparse import ArgumentParser

import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

import time
import numpy as np

PRINTS = ''

parser = ArgumentParser()
parser.add_argument('pose_config', help='Config file for pose')
parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
parser.add_argument('--video-path', type=str, help='Video path')
parser.add_argument('--show', action='store_true', default=False,
                    help='whether to show visualizations.')
parser.add_argument('--out-video-root', default='',
                    help='Root of the output video file. '
                    'Default not saving the visualization video.')
parser.add_argument('--device', default='cpu',
                    help='Device used for inference')
parser.add_argument('--box-thr', type=float, default=0.3,
                    help='Bounding box score threshold')
parser.add_argument('--kpt-thr', type=float, default=0.3,
                    help='Keypoint score threshold')
parser.add_argument('--on_cluster', type=int, default=0)
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--only_box', type=bool, default=False)
# parser.add_argument('--csv-path', type=str, help='CSV path')

args = parser.parse_args()

ON_CLUSTER = args.on_cluster
if ON_CLUSTER:
    print('NOTHING WILL PRINT, ON CLUSTER')


if ON_CLUSTER:
    def print(obj):
        global PRINTS
        PRINTS += obj + '\n'


def box_check(img, device='cpu'):
    global PRINTS, ON_CLUSTER
    flip = False
    det_config = '/home/filipkr/Documents/xjob/mmpose/mmdetection/' +\
        'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_model = '/home/filipkr/Documents/xjob/mmpose/mmdetection/' +\
        'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    det_model = init_detector(det_config, det_model, device=device)
    print('loaded detection model')

    det_results = inference_detector(det_model, img)
    # bbox = det_results[0]
    bbox = np.expand_dims(np.array(det_results[0])[0, :], axis=0)
    bbox[0, 2:4] = bbox[0, 2:4] + 100
    # print(bbox)
    if abs(bbox[0, 0] - bbox[0, 2]) > abs(bbox[0, 1] - bbox[0, 3]):
        flip = True
        bbox[0, 1] -= 100
        bbox = [[bbox[0, 1], bbox[0, 0], bbox[0, 3], bbox[0, 2], bbox[0, 4]]]
        print('frames will be flipped')
    else:
        bbox[0, 0] -= 100

    print('bounding box found: {0}'.format(bbox))
    if not ON_CLUSTER:
        show_result_pyplot(det_model, img, det_results)

    return bbox, flip


def flip_box(bbox, width):
    global PRINTS

    print(bbox)
    print(width)
    bbox[0, 0] = width - bbox[0, 0]
    bbox[0, 2] = width - bbox[0, 2]

    return bbox


def loop(args, rotate, fname, person_bboxes, pose_model, flipped=False):
    global PRINTS, ON_CLUSTER

    cap = cv2.VideoCapture(args.video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rotate:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    m_dim = max(size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(fname, fourcc, fps, size)
    poses = np.zeros((frames,
                      pose_model.cfg.channel_cfg['num_output_channels'], 2))
    dataset = pose_model.cfg.data['test']['type']

    skip_ratio = 1

    lmin = 1
    lmax = 0
    rmin = 1
    rmax = 0

    frame = 0
    t0 = time.perf_counter()
    prev_pose = 0
    while (cap.isOpened()):
        t1 = time.perf_counter()
        flag, img = cap.read()
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if flipped:
            img = cv2.flip(img, 1)
        if not flag:
            break

        # if frame > 66:
        # check every nd frame
        if frame % skip_ratio == 0:
            # test a single image, with a list of bboxes.
            pose_results = inference_top_down_pose_model(pose_model, img,
                                                         person_bboxes,
                                                         bbox_thr=args.box_thr,
                                                         format='xyxy',
                                                         dataset=dataset)
            t = time.perf_counter()

            print('Frame {0} out of {1} '.format(frame, frames) +
                  'analysed in {0} secs. '.format(t - t1) +
                  'Total time: {0} secs'.format(t - t0))

            # show the results
            if np.shape(pose_results)[0] > 0:
                prev_pose = pose_results

                ratios = pose_results[0]['keypoints'][:, 0:2] / m_dim

                lmin = min((ratios[13, 1], lmin))
                lmax = max((ratios[13, 1], lmax))
                rmin = min((ratios[14, 1], rmin))
                rmax = max((ratios[14, 1], rmax))

                if not flipped and ((rmax - rmin) > 0.1 or
                                    (frame > 150 and
                                     (rmax - rmin) > (lmax - lmin))):
                    # flipped = True
                    print('Left knee evaluated, restarting ' +
                          'with flipped images...')
                    cap.release()
                    videoWriter.release()
                    cv2.destroyAllWindows()
                    loop(args, rotate, fname, flip_box(person_bboxes, size[0]),
                         pose_model, True)
                    return

                poses[frame, ...] = ratios

            else:
                pose_results = prev_pose  # or maybe just skip saving
                print('lol')

        else:
            pose_results = prev_pose

        vis_img = vis_pose_result(pose_model, img, pose_results,
                                  dataset=dataset, kpt_score_thr=args.kpt_thr,
                                  show=False)

        if args.show or frame % skip_ratio == 0 and not ON_CLUSTER:
            cv2.imshow('Image', vis_img)

        # if save_out_video:
        videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame += 1

    cap.release()
    # if save_out_video:
    videoWriter.release()
    out_file = fname.replace('.mp4', '.npy')
    np.save(out_file, poses)

    cv2.destroyAllWindows()


def start(args):
    global PRINTS

    cap = cv2.VideoCapture(args.video_path)
    print('loaded video...')
    print('checking orientation and position')

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps)

    flag, img = cap.read()
    cap.release()
    person_bboxes, rotate = box_check(img)

    print(args.only_box)
    if args.only_box:
        # cv2.waitKey(0)
        return

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True
        print('save path: {0}'.format(args.out_video_root))

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 device=args.device)
    print('loaded pose model')

    dataset = pose_model.cfg.data['test']['type']
    print(dataset)

    mod_used = pose_model.cfg.model['backbone']['type']
    print('model used {0}'.format(mod_used))

    if save_out_video:
        if args.file_name == '':
            fname = os.path.join(args.out_video_root,
                                 f'vis_{os.path.basename(args.video_path)}')
            fname = fname.replace(fname[fname.find('.', -5)::], '')
            fname += str(int(np.round(fps))) + mod_used + dataset + '.mp4'
            print('FN {0}'.format(fname))
            while os.path.isfile(fname):
                fname = fname.replace('.mp4', '')

                idx = fname.find('-', -4) + 1
                if idx == 0:
                    fname += '-0.mp4'
                else:
                    fname = fname[:idx] + \
                        str(int(fname[idx::]) + 1) + '.mp4'
        else:
            fname = os.path.join(args.out_video_root, args.file_name)

        print(fname)

    loop(args, rotate, fname, person_bboxes, pose_model)


def main():
    global PRINTS, ON_CLUSTER
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    # parser = ArgumentParser()
    # parser.add_argument('pose_config', help='Config file for pose')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    # parser.add_argument('--video-path', type=str, help='Video path')
    # parser.add_argument('--show', action='store_true', default=False,
    #                     help='whether to show visualizations.')
    # parser.add_argument('--out-video-root', default='',
    #                     help='Root of the output video file. '
    #                     'Default not saving the visualization video.')
    # parser.add_argument('--device', default='cpu',
    #                     help='Device used for inference')
    # parser.add_argument('--box-thr', type=float, default=0.3,
    #                     help='Bounding box score threshold')
    # parser.add_argument('--kpt-thr', type=float, default=0.3,
    #                     help='Keypoint score threshold')
    # parser.add_argument('--file_name', type=str, default='')
    # parser.add_argument('--only_box', type=bool, default=False)
    # # parser.add_argument('--csv-path', type=str, help='CSV path')
    #
    # args = parser.parse_args()
    assert args.show or (args.out_video_root != '')

    # print('lollol')
    # print('lkjfds')
    start(args)
    if ON_CLUSTER:
        print_file = open(args.out_video_root + '/print_file.txt', 'w')
        print_file.write(PRINTS)


if __name__ == '__main__':
    # global PRINTS
    # print('starting...')
    main()
