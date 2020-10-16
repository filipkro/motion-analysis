import numpy as np
import cv2
from argparse import ArgumentParser
import filt


def create_video(cap, poses, video_writer):

    frame = 0
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break

        for pose in poses[frame, ...]:
            img = cv2.drawMarker(img, (int(np.round(pose[0])),
                                       int(np.round(pose[1]))), (255, 0, 0),
                                 markerType=cv2.MARKER_CROSS, markerSize=10,
                                 thickness=5)
        video_writer.write(img)
        frame += 1

    video_writer.release()
    print('done')


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    parser.add_argument('vid_file', help='Video file to add markers to')
    args = parser.parse_args()
    poses = np.load(args.np_file)
    cap = cv2.VideoCapture(args.vid_file)
    out_video = args.np_file.replace('.npy', '-filtered-2.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video, fourcc, fps, size)

    poses = poses[:, :, 0:2]
    if poses[0, 0, 0] < 1:
        poses[..., 0] = poses[..., 0] * 1920
        poses[..., 1] = poses[..., 1] * 1080

    poses = filt.outliers(poses)
    poses = filt.smoothing(poses)
    poses = filt.fix_hip(poses)

    create_video(cap, poses, video_writer)


if __name__ == '__main__':
    main()
