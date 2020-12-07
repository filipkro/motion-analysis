import cv2
import os
from argparse import ArgumentParser


def fix_folder(args):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for vid in os.listdir(args.vid):
        if vid.endswith(str(args.old) + '.mp4'):
            vid_path = os.path.join(args.vid, vid)
            cap = cv2.VideoCapture(vid_path)
            fname = vid_path.replace('50', '25')
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            videoWriter = cv2.VideoWriter(fname, fourcc, args.new_fps, size)
            while (cap.isOpened()):
                flag, img = cap.read()

                if not flag:
                    break

                videoWriter.write(img)

            cap.release()
            videoWriter.release()

            print(fname + 'done')

    print('DONE')


def main(args):
    if args.old != '':
        fix_folder(args)
    else:
        cap = cv2.VideoCapture(args.vid)
        fname = args.vid.replace('50', '25')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fname)
        videoWriter = cv2.VideoWriter(fname, fourcc, args.new_fps, size)

        while (cap.isOpened()):
            flag, img = cap.read()

            if not flag:
                break

            videoWriter.write(img)

        cap.release()
        videoWriter.release()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('vid', type=str)
    parser.add_argument('new_fps', type=int)
    parser.add_argument('--old', type=int, default=0)
    args = parser.parse_args()
    main(args)
