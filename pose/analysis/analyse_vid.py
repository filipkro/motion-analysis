import os
from argparse import ArgumentParser, ArgumentTypeError

import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

import time
import numpy as np


def box_check(img, folder_box, show_box=False, device='cpu'):
    flip = False
    # det_config = '/home/filipkr/Documents/xjob/mmpose/mmdetection/' +\
    #    'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_config = folder_box +\
        'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_model = folder_box +\
        'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    det_model = init_detector(det_config, det_model, device=device)
    print('loaded detection model')
    det_results = inference_detector(det_model, img)
    # bbox = det_results[0]
    bbox = np.expand_dims(np.array(det_results[0])[0, :], axis=0)
    bbox[0, 2:4] = bbox[0, 2:4] + 100
    bbox[0, 4] = 1
    # print(bbox)
    if abs(bbox[0, 0] - bbox[0, 2]) > abs(bbox[0, 1] - bbox[0, 3]):
        flip = True
        bbox[0, 1] -= 100
        bbox = [[bbox[0, 1], bbox[0, 0], bbox[0, 3], bbox[0, 2], bbox[0, 4]]]
        print('frames will be flipped')
    else:
        bbox[0, 0] -= 100

    print('bounding box found: {0}'.format(bbox))
    if show_box:
        show_result_pyplot(det_model, img, det_results)

    bbox = np.array(bbox)

    return bbox, flip


def check_pose4_flip180(pose_model, img, rotate, bbox, args, size):
    dataset = pose_model.cfg.data['test']['type']

    print(bbox)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    pose = inference_top_down_pose_model(pose_model, img, bbox,
                                         bbox_thr=args.box_thr,
                                         format='xyxy',
                                         dataset=dataset)

    if np.shape(pose)[0] > 0:
        if pose[0]['keypoints'][6, 1] > pose[0]['keypoints'][16, 1]:
            new_box = np.zeros((1, 5))
            new_box[0, 0] = size[0] - bbox[0, 2]
            new_box[0, 1] = size[1] - bbox[0, 3]
            new_box[0, 2] = size[0] - bbox[0, 0]
            new_box[0, 3] = size[1] - bbox[0, 1]
            new_box[0, 4] = bbox[0, 4]
            return True, new_box
    return False, bbox


def flip_box(bbox, width):
    '''
        flip boxes when evaluating left leg
    '''
    print(bbox)
    print(width)

    bbox[0, 0] = width - bbox[0, 0]
    bbox[0, 2] = width - bbox[0, 2]

    return bbox


def loop(args, rotate, fname, bbox, pose_model, flipped=False,
         rotate_180=False, t0=time.perf_counter()):

    cap = cv2.VideoCapture(args.video_path)

    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rotate:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.fname_format and args.flip2right:
        if os.path.basename(fname).split('-')[2] == 'L':
            flipped = True
            bbox = flip_box(bbox, size[0])

    m_dim = max(size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(fname, fourcc, fps, size)
    poses = np.zeros((frames,
                      pose_model.cfg.channel_cfg['num_output_channels'], 2))
    dataset = pose_model.cfg.data['test']['type']

    # skip_ratio = 1

    lmin = 1
    lmax = 0
    rmin = 1
    rmax = 0

    frame = 0
    # t0 = time.perf_counter()
    prev_pose = 0
    while (cap.isOpened()):
        t1 = time.perf_counter()
        flag, img = cap.read()
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if flipped:
            img = cv2.flip(img, 1)
        if not flag:
            break

        # if frame > 66:
        # check every nd frame
        if frame % args.skip_rate == 0:
            # test a single image, with a list of bboxes.
            pose_results = inference_top_down_pose_model(pose_model, img,
                                                         bbox,
                                                         bbox_thr=args.box_thr,
                                                         format='xyxy',
                                                         dataset=dataset)
            t = time.perf_counter()

            print('Frame {:.4f} out of {:.4f} '.format(frame, frames) +
                  'analysed in {:.4f} secs. '.format(t - t1) +
                  'Total time: {:.4f} secs'.format(t - t0))

            # show the results
            if np.shape(pose_results)[0] > 0:
                prev_pose = pose_results

                ratios = pose_results[0]['keypoints'][:, 0:2] / m_dim

                lmin = min((ratios[13, 1], lmin))
                lmax = max((ratios[13, 1], lmax))
                rmin = min((ratios[14, 1], rmin))
                rmax = max((ratios[14, 1], rmax))

                if args.allow_flip and not flipped and ((rmax - rmin) > 0.1 or
                                                        (frame > 150 and
                                                         (rmax - rmin) >
                                                         (lmax - lmin))):
                    # flipped = True
                    print('Left knee evaluated, restarting ' +
                          'with flipped images...')
                    cap.release()
                    videoWriter.release()
                    cv2.destroyAllWindows()
                    poses, meta, path = loop(args, rotate, fname,
                                             flip_box(bbox, size[0]),
                                             pose_model, flipped=True, t0=t0)
                    return poses, meta, path

                poses[frame, ...] = pose_results[0]['keypoints'][:, 0:2] \
                    if args.save_pixels else ratios

            else:
                pose_results = prev_pose  # or maybe just skip saving
                print('lol')

        else:
            pose_results = prev_pose

        vis_img = vis_pose_result(pose_model, img, pose_results,
                                  dataset=dataset, kpt_score_thr=args.kpt_thr,
                                  show=False)

        if args.show and frame % args.skip_rate == 0:
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

    name = os.path.basename(args.video_path).split('_')[0]
    # meta[name]
    meta = {'w': size[0], 'h': size[1], 'fps': fps}

    if args.save4_3d:
        meta_d = {}
        meta_d['video_metadata'] = {}

        meta_d = {'layout_name': 'coco', 'num_joints': 17,
                  'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],
                                         [2, 4, 6, 8, 10, 12, 14, 16]],
                  'video_metadata': {name: meta}}
        poses_3d = {}
        poses_3d['custom'] = [poses.astype('float32')]
        meta_poses = {}
        meta_poses[name] = poses_3d
        output_prefix_2d = 'data_2d_custom_'
        file_3d_name = args.out_video_root + output_prefix_2d \
            + os.path.basename(args.video_path)
        print(file_3d_name)
        np.savez_compressed(file_3d_name, positions_2d=meta_poses,
                            metadata=meta_d)

    return poses, meta, os.path.basename(args.video_path)
    # return time.perf_counter() - t0


def start(args):
    print(args.video_path)
    cap = cv2.VideoCapture(args.video_path)
    print('loaded video...')
    print('checking orientation and position')

    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))

    print('Frame rate: {} fps'.format(fps))

    flag, img = cap.read()

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

    orig_fname = os.path.basename(args.video_path)
    if args.fname_format:
        subject = orig_fname[0:2]
        action = orig_fname[2:5]
        leg = orig_fname[6]

        # if args.flip2right and leg == 'L':
        #     img =

        if save_out_video:
            fname = os.path.join(args.out_video_root, subject + '-' + action +
                                 '-' + leg + '-' + str(fps) + '.mp4')
    else:
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

    bbox, rotate = box_check(
        img, args.folder_box, device=args.device, show_box=args.show_box)

    if rotate:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    cap.release()

    rotate_180, bbox = check_pose4_flip180(pose_model, img, rotate,
                                           bbox, args, size)

    return loop(args, rotate, fname, bbox, pose_model, rotate_180=rotate_180)


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
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--show', type=str2bool, nargs='?', const=True,
                        default=False, help="show results.")
    parser.add_argument('--out-video-root', default='',
                        help='Root of the output video file. '
                        'Default not saving the visualization video.')
    parser.add_argument('--device', default='cpu',
                        help='Device used for inference')
    parser.add_argument('--box-thr', type=float, default=0.1,
                        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='Keypoint score threshold')
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--only_box', type=str2bool, nargs='?', const=True,
                        default=False, help="only show bounding box")
    parser.add_argument('--folder_box', type=str, default='')
    parser.add_argument('--show_box', type=str2bool, nargs='?', const=True,
                        default=False, help="show bounding box.")
    parser.add_argument('--allow_flip', type=str2bool, nargs='?',  # const=True,
                        default=False, help='for FL')
    parser.add_argument('--save_pixels', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='saveposes as pixels or ratio of im')
    parser.add_argument('--save4_3d', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='save poses along with meta data for 3d')
    parser.add_argument('--flip2right', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='flips video if name contains L')
    parser.add_argument('--fname_format', type=str2bool, nargs='?',
                        default=True,
                        help='if filename has format of marked videos')
    parser.add_argument('--skip_rate', type=int, default=1)

    args = parser.parse_args()

    if not args.fname_format:
        args.flip2right = False

    if args.flip2right:
        args.allow_flip = False

    print('format', args.fname_format)
    print('flip', args.flip2right)

    start(args)


if __name__ == '__main__':
    print('starting...')
    main()
