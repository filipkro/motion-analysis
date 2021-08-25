import os
import sys

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

BASE = os.path.abspath(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), os.pardir))

sys.path.append(os.path.join(BASE, 'mmpose'))
sys.path.append(os.path.join(BASE, 'mmdetection'))

from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1
    return bbox_xywh


def _xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0] - 1
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1] - 1
    return bbox_xyxy


def _box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
        else:
            results['image_file'] = ''
        img = mmcv.imread(results['img_or_path'], self.color_type,
                          self.channel_order)
        results['img'] = img
        return results


def _inference_single_pose_model(model, img_or_path, bbox, dataset):
    """Inference a single bbox.

    num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        image_name (str | np.ndarray):Image_name
        bbox (list | np.ndarray): Bounding boxes (with scores),
            shaped (4, ) or (5, ). (left, top, width, height, [score])
        dataset (str): Dataset name.

    Returns:
        ndarray[Kx3]: Predicted pose x, y, score.
    """
    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    assert len(bbox) in [4, 5]
    center, scale = _box2cs(cfg, bbox)

    flip_pairs = None
    if dataset == 'TopDownCocoDataset' or dataset == 'TopDownOCHumanDataset':
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]
    elif dataset == 'TopDownCocoWholeBodyDataset':
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]
        flip_pairs = body + foot + face + hand
    elif dataset == 'TopDownAicDataset':
        flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
    elif dataset == 'TopDownOneHand10KDataset' or \
            dataset == 'TopDownFreiHandDataset':
        flip_pairs = []
    elif dataset == 'TopDownMpiiDataset':
        flip_pairs = []
    else:
        raise NotImplementedError()

    # prepare data
    data = {
        'img_or_path':
        img_or_path,
        'center':
        center,
        'scale':
        scale,
        'bbox_score':
        bbox[4] if len(bbox) == 5 else 1,
        'dataset':
        dataset,
        'joints_3d':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'joints_3d_visible':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'rotation':
        0,
        'ann_info': {
            'image_size': cfg.data_cfg['image_size'],
            'num_joints': cfg.data_cfg['num_joints'],
            'flip_pairs': flip_pairs
        }
    }
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'].data[0]

    # forward the model
    with torch.no_grad():
        #print('im')
        #print(data['img'].shape)
        # im2show = np.squeeze(np.array(data['img']), axis=0)
        # plt.imshow(np.moveaxis(im2show, 0, -1))
        # plt.show()
        all_preds, _, _ = model(
            return_loss=False, img=data['img'], img_metas=data['img_metas'])
    # print(all_preds)
    return all_preds[0]


def inference_top_down_pose_model(model,
                                  img_or_path,
                                  person_bboxes,
                                  bbox_thr=None,
                                  format='xywh',
                                  dataset='TopDownCocoDataset'):
    """Inference a single image with a list of person bounding boxes.

    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        image_name (str| np.ndarray): Image_name
        person_bboxes: (np.ndarray[P x 4] or [P x 5]): Each person bounding box
            shaped (4, ) or (5, ), contains 4 box coordinates (and score).
        bbox_thr: Threshold for bounding boxes. Only bboxes with higher scores
            will be fed into the pose detector. If bbox_thr is None, ignore it.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.

    Returns:
        list[dict]: The bbox & pose info.

            Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score])
            and the pose (ndarray[Kx3]): x, y, score
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']
    # transform the bboxes format to xywh
    if format == 'xyxy':
        person_bboxes = _xyxy2xywh(np.array(person_bboxes))
    pose_results = []

    if len(person_bboxes) > 0:
        if bbox_thr is not None:
            person_bboxes = person_bboxes[person_bboxes[:, 4] > bbox_thr]
        for bbox in person_bboxes:
            pose = _inference_single_pose_model(model, img_or_path, bbox,
                                                dataset)
            pose_results.append({
                'bbox':
                _xywh2xyxy(np.expand_dims(np.array(bbox), 0)),
                'keypoints':
                pose,
            })

    return pose_results


def vis_pose_result(model,
                    img,
                    result,
                    kpt_score_thr=0.3,
                    dataset='TopDownCocoDataset',
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if dataset == 'TopDownCocoDataset' or dataset == 'BottomUpCocoDataset' \
            or dataset == 'TopDownOCHumanDataset':
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
        ]]

    elif dataset == 'TopDownCocoWholeBodyDataset':
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [16, 18],
                    [16, 19], [16, 20], [17, 21], [17, 22], [17, 23], [24, 25],
                    [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31],
                    [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37],
                    [37, 38], [38, 39], [39, 40], [92, 93], [93, 94], [94, 95],
                    [95, 96], [92, 97], [97, 98], [98, 99], [99, 100],
                    [92, 101], [101, 102], [102, 103], [103, 104], [92, 105],
                    [105, 106], [106, 107], [107, 108], [92, 109], [109, 110],
                    [110, 111], [111, 112], [113, 114], [114, 115], [115, 116],
                    [116, 117], [113, 118], [118, 119], [119, 120], [120, 121],
                    [113, 122], [122, 123], [123, 124], [124, 125], [113, 126],
                    [126, 127], [127, 128], [128, 129], [113, 130], [130, 131],
                    [131, 132], [132, 133]]

        pose_limb_color = palette[
            [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16] +
            [
                16,
            ] * 6 + [
                19,
            ] * 16 + [
                19,
            ] * 40]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] + [
                0,
            ] * 6 + [
                19,
            ] * 68 + [
                19,
            ] * 42]

    elif dataset == 'TopDownAicDataset':
        skeleton = [[3, 2], [2, 1], [1, 14], [14, 4], [4, 5], [5, 6], [9, 8],
                    [8, 7], [7, 10], [10, 11], [11, 12], [13, 14], [1, 7],
                    [4, 10]]

        pose_limb_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
        ]]
        pose_kpt_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
        ]]

    elif dataset == 'TopDownOneHand10KDataset' or \
            dataset == 'TopDownFreiHandDataset':
        skeleton = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8],
                    [8, 9], [1, 10], [10, 11], [11, 12], [12, 13], [1, 14],
                    [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20],
                    [20, 21]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]
    elif dataset == 'TopDownMpiiDataset':

        skeleton = [[1, 2], [2, 3], [3, 7], [4, 7], [4, 5], [5, 6], [7, 8],
                    [8, 9], [9, 10], [8, 13], [13, 12], [12, 11], [8, 14],
                    [14, 15], [15, 16]]

        pose_limb_color = palette[[
            0, 0, 7, 7, 0, 0, 7, 7, 7, 9, 16, 16, 9, 16, 16
        ]]

        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0
        ]]
    else:
        raise NotImplementedError()

    img = model.show_result(
        img,
        result,
        skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        show=show,
        out_file=out_file)

    return img
