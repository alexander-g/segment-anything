import argparse
from distutils.version import LooseVersion
import os
import re
import time
import typing as tp
import warnings

import sys
sys.path.append('.')

import numpy as np
import onnx
import onnxruntime
import PIL.Image
import torch


from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from scripts.export_onnx_model import run_export as run_onnx_export


class SamEncoder(torch.nn.Module):
    def __init__(self, model_type:str, checkpoint:str):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder = sam.image_encoder
        self.mean = sam.pixel_mean
        self.std  = sam.pixel_std
        self.size = sam.image_encoder.img_size
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.uint8 and x.ndim == 3 and x.shape[-1] == 3, \
            'Input in uint8 HWC format expected'
        x = x.permute(2,0,1)
        x = preprocess(x, self.mean, self.std, self.size)
        ft = self.image_encoder(x)
        return ft


def get_preprocess_shape(oldh:torch.Tensor, oldw:torch.Tensor, long_side_length:int) -> torch.Tensor:
    # copied from utils/transforms.py
    scale = long_side_length * 1.0 / torch.maximum(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = (neww + 0.5).long()
    newh = (newh + 0.5).long()
    return torch.stack([newh, neww])


def preprocess(
    x:          torch.Tensor, 
    pixel_mean: torch.Tensor, 
    pixel_std:  torch.Tensor, 
    img_size:   int,
) -> torch.Tensor:
    '''Normalize pixel values and pad to a square input.'''

    h = torch.tensor(x.shape[-2])
    w = torch.tensor(x.shape[-1])
    new_size = get_preprocess_shape(h, w, img_size)
    new_h    = int(new_size[0])
    new_w    = int(new_size[1])
    x = torch.nn.functional.interpolate(
        x[None], 
        (new_h, new_w), 
        mode          = "bilinear", 
        align_corners = False, 
        antialias     = True,
    )[0]

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = torch.nn.functional.pad(x, [0, padw, 0, padh])[None]
    return x



class DecoderPreprocessor(torch.nn.Module):
    def __init__(self, inputsize:int):
        super().__init__()
        self.inputsize = inputsize

    def forward(self, box:torch.Tensor, orig_im_size:torch.Tensor):
        assert box.shape == (4,)
        assert orig_im_size.shape == (2,)
        
        # re-using orig_im_size from the og graph, height first width second
        h = orig_im_size.to(torch.float32)[0]
        w = orig_im_size.to(torch.float32)[1]
        new_size = get_preprocess_shape(h, w, self.inputsize)
        new_h    = new_size[0]
        new_w    = new_size[1]

        coords_ = box.reshape(-1,2).float()
        coords = torch.stack([
            coords_[..., 0] * (new_w / w),
            coords_[..., 1] * (new_h / h),
        ], dim=-1)

        labels = torch.tensor([2,3], device=box.device).float()
        labels = labels[None,:]

        return {
            'point_coords': coords,
            'point_labels': labels,
            'orig_im_size_passthrough': torch.tensor(orig_im_size),
            'mask_input':     torch.zeros((1, 1, 256, 256)),
            'has_mask_input': torch.zeros(1),
        }

def export_preprocessor(dst:str, preprocessor:DecoderPreprocessor):
    dummy_inputs = {
        'box':          torch.ones(4),
        'orig_im_size': torch.ones(2),
    }
    output_names = [
        'point_coords', 
        'point_labels', 
        'orig_im_size_passthrough',
        'mask_input',
        'has_mask_input',
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(dst, "wb") as f:
            torch.onnx.export(
                preprocessor,
                tuple(dummy_inputs.values()),
                f=f,
                export_params = False,
                verbose       = False,
                opset_version = 17,
                do_constant_folding = False,
                input_names  = list(dummy_inputs.keys()),
                output_names = output_names,
            )
    return dst


class DecoderPostprocessor(torch.nn.Module):
    def forward(self, masks:torch.nn.Module):
        return masks[0,0] > 0.0

def export_postprocessor(dst:str, postprocessor:DecoderPostprocessor):
    dummy_inputs = {
        'masks_in': torch.ones([1,1,100,100]),
    }
    output_names = [
        'masks',
    ]
    dynamic_axes = {
        "masks_in": {2: "height", 3:"width"},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(dst, "wb") as f:
            torch.onnx.export(
                postprocessor,
                tuple(dummy_inputs.values()),
                f=f,
                export_params = False,
                verbose       = False,
                opset_version = 17,
                do_constant_folding = False,
                input_names  = list(dummy_inputs.keys()),
                output_names = output_names,
                dynamic_axes = dynamic_axes,
            )
    return dst



def modify_onnx_decoder(onnx_decoder_path:str, inputsize:int):
    decoder_onnx = onnx.load(onnx_decoder_path)
    preprocessor = DecoderPreprocessor(inputsize)
    onnx_preprocessor_path = \
        export_preprocessor(onnx_decoder_path+'.preprocessor.onnx', preprocessor)
    preprocessor_onnx = onnx.load(onnx_preprocessor_path)

    
    io_map = [
        ('point_coords', 'point_coords'), 
        ('point_labels', 'point_labels'),
        ('orig_im_size_passthrough', 'orig_im_size'),
        ('mask_input',     'mask_input'),
        ('has_mask_input', 'has_mask_input'),
    ]

    merged = onnx.compose.merge_models(
        preprocessor_onnx, 
        decoder_onnx, 
        io_map,
        outputs = ['masks'],
        prefix1 = 'pre_',
    )

    print([i.name for i in merged.graph.input])
    print([o.name for o in merged.graph.output])
    print()

    postprocessor = DecoderPostprocessor()
    onnx_postprocessor_path = \
        export_postprocessor(onnx_decoder_path+'.postprocessor.onnx', postprocessor)
    postprocessor_onnx = onnx.load(onnx_postprocessor_path)

    io_map = [
        ('masks', 'masks_in'),
    ]

    merged2 = onnx.compose.merge_models(
        merged, 
        postprocessor_onnx, 
        io_map,
        outputs = ['post_masks'],
        prefix2 = 'post_',
    )
    print([i.name for i in merged2.graph.input])
    print([o.name for o in merged2.graph.output])
    print(merged2.graph.output)

    onnx.save(merged2, onnx_decoder_path)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', 
        required = True,
        help     = 'Path to .pth SAM checkpoint'
    )
    parser.add_argument(
        '--modeltype',  
        required = True,
        help     = 'Type of model e.g. vit_b',
    )
    parser.add_argument(
        '--outputdir',
        required = True,
        help     = 'Where to save the models'
    )
    return parser


def check_onnx_version(version:str, required:str):
    assert LooseVersion(version) >= LooseVersion(required), \
        f'ONNx version {required} needed'

if __name__ == '__main__':
    check_onnx_version(onnx.__version__, required='1.19.0')

    args = get_argparser().parse_args()

    sam_encoder = SamEncoder(args.modeltype, args.checkpoint)
    sam_encoder_ts = torch.jit.script(sam_encoder)

    os.makedirs(args.outputdir, exist_ok = True)
    encoder_dst = os.path.join(
        args.outputdir, 
        f'sam_encoder_{args.modeltype}.torchscript'
    )
    decoder_dst = os.path.join(
        args.outputdir,
        f'sam_decoder_{args.modeltype}.onnx'
    )

    run_onnx_export(
        model_type = args.modeltype,
        checkpoint = args.checkpoint,
        output     = decoder_dst,
        opset      = 17,
        return_single_mask   = True
    )
    modify_onnx_decoder(decoder_dst, sam_encoder.size)

    sam_encoder_ts.save(encoder_dst)

    print('done')

