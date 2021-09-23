from mmcv.utils import Registry, build_from_cfg
PRIOR_GENERATORS = Registry('Generator for anchors and points')
ANCHOR_GENERATORS = PRIOR_GENERATORS


def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)
