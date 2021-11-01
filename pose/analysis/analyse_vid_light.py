import os
import sys
from argparse import ArgumentParser, ArgumentTypeError

import cv2
import time
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

sys.path.append(os.path.join(BASE, 'mmpose'))
sys.path.append(os.path.join(BASE, 'mmpose/mmdetection'))


def box_check(img, folder_box, show_box=False, device='cpu'):
    ''' Checks whether person is standing upright or not in video '''
    from mmdet.apis import inference_detector, init_detector

    flip = False

    det_config = folder_box + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_model = folder_box + '/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    det_model = init_detector(det_config, det_model, device=device)
    print('loaded detection model')
    det_results = inference_detector(det_model, img)
    del det_model

    bbox = np.expand_dims(np.array(det_results[0])[0, :], axis=0)
    bbox[0, 2:4] = bbox[0, 2:4] + 100
    bbox[0, 4] = 1

    if abs(bbox[0, 0] - bbox[0, 2]) > abs(bbox[0, 1] - bbox[0, 3]):
        flip = True
        bbox[0, 1] -= 100
        bbox = [[bbox[0, 1], bbox[0, 0], bbox[0, 3], bbox[0, 2], bbox[0, 4]]]
        print('frames will be flipped')
    else:
        bbox[0, 0] -= 100

    bbox = np.array(bbox)

    return bbox, flip


def check_pose4_flip180(img, rotate, bbox, args, size, pose_config,
                        pose_checkpoint, device='cpu'):
    ''' checks if person is upside down or not '''
    from mmpose.apis import inference_top_down_pose_model, init_pose_model

    pose_model = init_pose_model(pose_config, pose_checkpoint,
                                 device=device)
    dataset = pose_model.cfg.data['test']['type']

    print(bbox)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    pose = inference_top_down_pose_model(pose_model, img, bbox,
                                         bbox_thr=args.box_thr,
                                         format='xyxy',
                                         dataset=dataset)
    del pose_model

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


def re_est_bbox(img, folder_box, flip90, flip180, flip2right, device='cpu'):
    ''' re-estimates bounding box with necessary rotations, adds 10% extra '''
    from mmdet.apis import inference_detector, init_detector

    det_config = folder_box + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_model = folder_box + '/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    if flip90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if flip180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    if flip2right:
        img = cv2.flip(img, 1)

    det_model = init_detector(det_config, det_model, device=device)
    det_results = inference_detector(det_model, img)
    del det_model
    bbox = np.expand_dims(np.array(det_results[0])[0, :], axis=0)
    dy = bbox[0, 3] - bbox[0, 1]
    dx = bbox[0, 2] - bbox[0, 0]
    bbox[0, 3] += 0.1 * dy
    bbox[0, 0] -= 0.1 * dx
    bbox[0, 2] += 0.1 * dx
    # bbox[0, 2:4] = bbox[0, 2:4] + 100
    bbox[0, 4] = 1

    return bbox


def flip_box(bbox, width):
    '''
        flip boxes when evaluating left leg
    '''
    print(bbox)
    print(width)

    bbox[0, 0] = width - bbox[0, 0]
    bbox[0, 2] = width - bbox[0, 2]

    return bbox


def loop(args, rotate, bbox, rotate_180=False, t0=time.perf_counter(),
         cap=None):

    from mmpose.apis import (inference_top_down_pose_model,
                             init_pose_model, vis_pose_result) # noqa

    if cap is None:
        cap = cv2.VideoCapture(args.video_path)

    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rotate:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print(size)

    m_dim = max(size)
    if args.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fname = os.path.basename(args.video_path).split('.')[0] + '-out.mp4'
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, size)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 device=args.device)
    poses = np.zeros((frames,
                      pose_model.cfg.channel_cfg['num_output_channels'], 2))
    dataset = pose_model.cfg.data['test']['type']

    frame = 0
    prev_pose = 0
    while (cap.isOpened()):
        t1 = time.perf_counter()
        flag, img = cap.read()

        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if args.flip:
            img = cv2.flip(img, 1)
        if not flag:
            break

        if True:
            # check every nd frame
            if frame % args.skip_rate == 0:
                # test a single image, with a list of bboxes.
                pose_results = inference_top_down_pose_model(pose_model, img,
                                                             bbox,
                                                             bbox_thr=args.box_thr,
                                                             format='xyxy',
                                                             dataset=dataset)

                t = time.perf_counter()

                print('Frame {:.0f} out of {:.0f} '.format(frame, frames) +
                      'analysed in {:.4f} secs. '.format(t - t1) +
                      'Total time: {:.4f} secs'.format(t - t0))

                # show the results
                if np.shape(pose_results)[0] > 0:
                    prev_pose = pose_results

                    poses[frame, ...] = pose_results[0]['keypoints'][:, 0:2] \
                        if args.save_pixels else \
                        pose_results[0]['keypoints'][:, 0:2] / m_dim

                else:
                    pose_results = prev_pose  # or maybe just skip saving
                    print('lol')

            else:
                pose_results = prev_pose

            vis_img = vis_pose_result(pose_model, img, pose_results,
                                      dataset=dataset,
                                      kpt_score_thr=args.kpt_thr, show=False)

            if args.show and frame % args.skip_rate == 0:
                print(f'args show: {args.show}')
                print(f'rest: {frame % args.skip_rate == 0}')
                cv2.imshow('Image', vis_img)

            if args.save_vid:
                if args.flip:  # flip to produce video as original
                    vis_img = cv2.flip(vis_img, 1)
                videoWriter.write(vis_img)
            if frame == 10:
                if args.flip:  # flip to produce video as original
                    vis_img = cv2.flip(vis_img, 1)
                img2save = cv2.resize(vis_img, (int(vis_img.shape[0]/5),
                                                int(vis_img.shape[1])),
                                      interpolation=cv2.INTER_AREA)
                cv2.imwrite('/app/debug.jpg', img2save)
                del img2save

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame += 1

    del pose_model

    cap.release()
    if args.save_vid:
        videoWriter.release()

    cv2.destroyAllWindows()

    meta = {'w': size[0], 'h': size[1], 'fps': fps}
    print(poses.shape)

    return poses, meta, os.path.basename(args.video_path)


def start(args):
    print(args.video_path)
    cap = cv2.VideoCapture(args.video_path)
    print(cap)
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('loaded video...')
    print('checking orientation and position')

    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
    print(fps)

    print('Frame rate: {} fps'.format(fps))
    flag, img = cap.read()

    # person upright or not:
    bbox, rotate = box_check(img, args.folder_box, device=args.device)

    if rotate:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    cap.release()
    # upside down or not:
    rotate_180, bbox = check_pose4_flip180(img, rotate, bbox, args, size,
                                           args.pose_config,
                                           args.pose_checkpoint)

    flip2right_leg = args.flip

    bbox = re_est_bbox(img, args.folder_box, rotate, rotate_180,
                       flip2right_leg, device=args.device)

    return loop(args, rotate, bbox, rotate_180=rotate_180)


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
    parser.add_argument('--show', type=str2bool, nargs='?',
                        default=False, help="show results.")
    parser.add_argument('--device', default='cpu',
                        help='Device used for inference')
    parser.add_argument('--box-thr', type=float, default=0.1,
                        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.1,
                        help='Keypoint score threshold')
    parser.add_argument('--folder_box', type=str, default='')
    parser.add_argument('--save_pixels', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='saveposes as pixels or ratio of im')
    parser.add_argument('--skip_rate', type=int, default=1)
    parser.add_argument('--flip', type=str2bool, default=False)
    parser.add_argument('--save_vid', type=str2bool, default=False)

    args = parser.parse_args()

    start(args)


if __name__ == '__main__':
    print('starting...')
    main()
