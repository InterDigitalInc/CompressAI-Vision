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
from compressai_vision.datasets import ImageFolder, SFUHW_ImageFolder, COCO_ImageFolder
from torchvision import transforms

from torch.utils.data import DataLoader

from tqdm import tqdm

# (fracape) WORK IN PROGRESS!
# probably need more modes and sub options about dumping results / tensors or not
MODES = [
    "full, network_first_part, network_second_part, feature_encode, feature_decode"
]

directory = os.getcwd()
MODELS={'faster_rcnn_R_50_FPN_3x': {'cfg': f'{directory}//compressai-fcvcm/models/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', 'weight': f'{directory}/compressai-fcvcm/weights/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl'},
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
    

    kargs = MODELS[args.model]
    model = faster_rcnn_X_101_32x8d_FPN_3x(device, **kargs)
    #model = mask_rcnn_X_101_32x8d_FPN_3x(device, **kargs)
    
    #test_dataset = SFUHW_ImageFolder(args.dataset_folder, model.cfg)

    from detectron2.data import DatasetCatalog, MetadataCatalog 
    
    dataset_name="mpeg-coco"

    test_dataset = COCO_ImageFolder(args.dataset_folder, model.cfg)
        #(dataset_name=dataset_name, args.dataset_folder, model.cfg)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        sampler= test_dataset.sampler,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    #temporary
    from detectron2.evaluation import COCOEvaluator
    evaluator = COCOEvaluator(dataset_name, model.get_cfg(), False, output_dir='./vision_output/', use_fast_impl=False)

    evaluator.reset()
    for d in tqdm(test_dataloader):
        #features = model.input_to_features(d)
        results = model(d)
        #print(type(results))

        evaluator.process(d, results)
       
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
