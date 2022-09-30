# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
# from .swin_mlp import SwinMLP


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
                                num_classes=config.MODEL.NUM_CLASSES
                                )
    
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
