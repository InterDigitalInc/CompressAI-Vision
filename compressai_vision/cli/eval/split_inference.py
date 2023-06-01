# Copyright (c) 2022-2023, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Compression Split inference: Compression of intermediate data:
   Feature Compression for Video Coding for Machines (FC-VCM pipeline)
"""

import os
from compressai_vision.model_wrappers import *
from compressai_vision.datasets import *
from torchvision import transforms

from torch.utils.data import DataLoader

from tqdm import tqdm

from compressai_vision.utils import readwriteYUV, PixelFormat

# (fracape) WORK IN PROGRESS!
# probably need more modes and sub options about dumping results / tensors or not
MODES = [
    "full, network_first_part, network_second_part, feature_encode, feature_decode"
]

directory = os.getcwd()
MODELS={'faster_rcnn_R_50_FPN_3x': {'cfg': f'{directory}/compressai-fcvcm/models/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', 'weight': f'{directory}/compressai-fcvcm/weights/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl'},
        'master_rcnn_R_50_FPN_3x': {'cfg': f'{directory}/compressai-fcvcm/models/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 'weight': f'{directory}/compressai-fcvcm/weights/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'},
        'faster_rcnn_X_101_32x8d_FPN_3x': {'cfg': f'{directory}/compressai-fcvcm/models/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', 'weight': f'{directory}/compressai-fcvcm/weights/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl'},
        'mask_rcnn_X_101_32x8d_FPN_3x': {'cfg': f'{directory}/compressai-fcvcm/models/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml', 'weight': f'{directory}/compressai-fcvcm/weights/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'},
        }

def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "split-inference",
        parents=parents,
        help="split inference scenario",
    )
    required_group = subparser.add_argument_group("required arguments")
    required_group.add_argument(
        "--dataset-folder",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset",
    )
    required_group.add_argument(
        "--model",
        action="store",
        type=str,
        required=True,
        default=None,
        #nargs="+",
        help="name of model.",
    )
    required_group.add_argument(
        "--mode",
        action="store",
        type=str,
        required=True,
        default="full",
        help="Part of the pipeline to run (default: %(default)s).",
    )
    required_group.add_argument(
        "--compression",
        action="store",
        type=str,
        required=True,
        default="full",
        nargs="+",
        help="Part of the pipeline to run (default: %(default)s).",
    )


def main(args):
    # check that only one is defined
    assert args.dataset_folder is not None, "please provide dataset name"
    assert args.model is not None, "please provide model name"

    device = 'cuda'

    #to get the current working directory
    rwYUV = readwriteYUV(device, format=PixelFormat.YUV400_10le, align=16)

    kargs = MODELS[args.model]
    #model = faster_rcnn_X_101_32x8d_FPN_3x(device, **kargs)
    model = mask_rcnn_X_101_32x8d_FPN_3x(device, **kargs)
    
    bitdepth = 10

    test_dataset = MPEGOIV6_ImageFolder(args.dataset_folder, model.cfg)
    packing_all_in_one = True
    #packing_all_in_one = False

    #test_dataset = SFUHW_ImageFolder(args.dataset_folder, model.cfg)
    #packing_all_in_one = True

    #test_dataset = COCO_ImageFolder(args.dataset_folder, model.cfg)
    #packing_all_in_one = False

    dataset_name = test_dataset.get_dataset_name()
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        sampler= test_dataset.sampler,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    def min_max_normalization(x, minv:float, maxv:float, bitdepth=8):
        max_num_bins = (2**bitdepth) - 1

        out = ((x - minv) / (maxv - minv)).clamp_(0, 1)
        mid_level = (-minv / (maxv - minv))

        return (out * max_num_bins).floor(), int(mid_level * max_num_bins + 0.5)

    def min_max_inv_normalization(x, minv:float, maxv:float, bitdepth=8):
        out = x / ((2**bitdepth) - 1)
        out = (out * (maxv - minv)) + minv
        return out
    
    #temporary
    from detectron2.evaluation import COCOEvaluator
    #evaluator = COCOEvaluator(dataset_name, model.get_cfg(), False, output_dir='./vision_output/', use_fast_impl=False)
    evaluator = COCOEvaluator(dataset_name, False, output_dir='./vision_output/', use_fast_impl=False)
    
    # Only for MPEG-OIV6
    deccode_compressed_rle(evaluator._coco_api.anns)
    
    evaluator.reset()
    setWriter = False
    setReader = False
    for e, d in enumerate(tqdm(test_dataloader)):
        org_img_size = {'height': d[0]['height'], 'width': d[0]['width']}

        features, input_img_size = model.input_to_feature_pyramid(d)

        frame, feature_size, subframe_height = model.reshape_feature_pyramid_to_frame(features, packing_all_in_one=packing_all_in_one)

        if packing_all_in_one:
            minv, maxv = test_dataset.get_min_max_across_tensors()
            normalized_frame,  mid_level = min_max_normalization(frame, minv, maxv, bitdepth=bitdepth)

            ## dump yuv
            #if setWriter is False:
            #    rwYUV.setWriter("/pa/home/hyomin.choi/Projects/compressai-fcvcm/out_tensor/test.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    #setWriter = True
            
            #rwYUV.write_single_frame(normalized_frame, mid_level=mid_level)

            # read yuv
            #if setReader is False:
            #    rwYUV.setReader("/mnt/wekamount/RI-Users/hyomin.choi/Projects/compressai-fcvcm/out_tensor/BasketballDrill.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    rwYUV.setReader("/pa/home/hyomin.choi/Projects/compressai-fcvcm/out_tensor/test.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    setReader = True

            #loaded_normalized_frame = rwYUV.read_single_frame(e)
            #normalized_frame = rwYUV.read_single_frame(0)

            #diff = normalized_frame - loaded_normalized_frame
            #if setWriter is False:
            #    rwYUV.setWriter("/pa/home/hyomin.choi/Projects/compressai-fcvcm/out_tensor/diff.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    setWriter = True
            
            #rwYUV.write_single_frame((diff+256), mid_level=mid_level)
    
            rescaled_frame = min_max_inv_normalization(normalized_frame, minv, maxv, bitdepth=bitdepth)
        else:
            rescaled_frame = frame
        
        back_to_features = model.reshape_frame_to_feature_pyramid(rescaled_frame, feature_size, subframe_height, packing_all_in_one=packing_all_in_one)

        results = model.feature_pyramid_to_output(back_to_features, org_img_size, input_img_size)

        #results = model(d)
        #print(type(results))

        evaluator.process(d, results)
    
    # temp
    total_annot = 0
    cate_list = []
    for key, cat_list in evaluator._coco_api.catToImgs.items():
        total_annot += len(cat_list)
        cate_list.append(key)
    print(len(cate_list), total_annot)

    results = evaluator.evaluate() 
    print(results)
    # (fracape) WORK IN PROGRESS!
    # get dataset, read folders of PNG files for now

    # if first_part:
    # run first part
    #
    #
    # if compression:
    # get / read intermediate features
    # for quality in qpars:
    # run feature compression
    # run decompression (can be conditional)

    # if second part
    # get / read intermediate features  (decompressed or original)
    # run second part
    #
    # get results / analyze results
