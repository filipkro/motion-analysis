from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
# from .detectors_resnext import DetectoRS_ResNeXt
# from .hourglass import HourglassNet
from .hrnet import HRNet
# from .regnet import RegNet
# from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
# from .resnext import ResNeXt
# from .ssd_vgg import SSDVGG

__all__ = ['ResNet', 'ResNetV1d', 'HRNet', 'DetectoRS_ResNet', 'Darknet'
]
