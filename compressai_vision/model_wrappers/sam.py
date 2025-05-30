from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import cv2
import numpy as np
import pandas
import csv
import os
import base64
from torch.nn import functional as F

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, ImageList

#sam = sam_model_registry["vit_h"](checkpoint="/t/vic/hevc_simulations/rosen/build/compressai13_sam/weights/sam/sam_vit_h_4b8939.pth")
#predictor = SamPredictor(sam)
#mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper

__all__ = [
    "sam_vit_h_4b8939",
    "sam_vit_b_01ec64",
    "sam_vit_l_0b3195",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")

class Boxes:
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __repr__(self):
        return f"Boxes(tensor({self.tensor}))"

class Instances:
    def __init__(self, image_size):
        self.image_height, self.image_width = image_size
        self._fields = {}

    def __setitem__(self, key, value):
        self._fields[key] = value

    def __getitem__(self, key):
        return self._fields[key]

    def get_fields(self):
        return self._fields

    def __len__(self):
        # Find the first field with a valid __len__ and use that
        for value in self._fields.values():
            try:
                return len(value)
            except TypeError:
                continue
        raise TypeError("No field with valid length found.")

    def __repr__(self):
        field_strs = [f"{k}: {v}" for k, v in self._fields.items()]
        fields_repr = ", ".join(field_strs)
        return (
            f"Instances(num_instances={len(self)}, "
            f"image_height={self.image_height}, image_width={self.image_width}, "
            f"fields=[{fields_repr}])"
        )




class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    ImageEncoder = "imgenc" #features output from neck.3.bias


class SAM(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        super().__init__(device)
      
        self.model = sam_model_registry["vit_h"](checkpoint=kwargs["weights"]).to(device).eval()
        self.model.load_state_dict(torch.load(kwargs["weights"]))

        self.backbone = self.model.image_encoder
        self.prompt_encoder = self.model.prompt_encoder
        self.head = self.model.mask_decoder


        #SamPredictor(self.model)
        #print(SamPredictor)
        self.supported_split_points = Split_Points

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()

        #if self.split_id == str(self.supported_split_points.ImageEncoder):
        self.split_layer_list = ["imgenc"]
        #else:
        #    raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        self.annotation_file = '/o/projects/proj-river/ctc_sequences/vcm_testdata/samtest/annotations/mpeg-oiv6-segmentation-coco_fortest.json'

    @property
    def SPLIT_IMGENC(self):
        return str(self.supported_split_points.ImageEncoder)


    def input_to_features(self, x, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""

        self.model = self.model.to(device).eval()
 
        if self.split_id == self.SPLIT_IMGENC:
            return self._input_to_image_encoder(x)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")

        raise NotImplementedError
        

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_IMGENC:
            return self._image_encoder_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        else:
            self.logger.error(f"Not supported split points {self.split_id}")

        raise NotImplementedError

    def postprocess_masks(masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...],) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
       

    @torch.no_grad()
    def _input_to_image_encoder(self, x):
        """Computes and return encoded image all the way from the input"""
        #TODO pre_processing
        #print("AAAAA _input_to_image_encoder", x ,'\n') 
        #imgs = ImageList(x)
        imgs = x[0]["image"]
        feature = {}
        feature["p"] = self.backbone(imgs)

        print("feature len", len(feature), type(feature))
        image_sizes = [x[0]["height"], x[0]["width"]] 
        #prompts = {}
        #prompts["points"] = [[469, 295]] 
        return {"data": feature, "input_size": image_sizes}


    @torch.no_grad()
    def get_input_size(self, x):
        """Computes input image size to the network"""
        imgs = ImageList(x)
        return imgs.image_sizes


    @torch.no_grad()
    def _image_encoder_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs  downstream task using the encoded image feature

        """

        # Replacing tag names for interfacing with NN-part2
        #x = dict(zip(self.features_at_splits.keys(), x.values()))
        #print("AAAAA x dict", x.keys())

        input_points = [[469, 295]] #prompts["points"]
        input_points = np.array(input_points)
        input_points_ = torch.tensor(input_points)
        input_points_ = input_points_.unsqueeze(-1)
        input_points_ = input_points_.permute(2, 0, 1)
        
        input_labels = np.array([1])
        input_labels_ = torch.tensor(input_labels)
        input_labels_ = input_labels_.unsqueeze(-1)
        input_labels_ = input_labels_.permute(1, 0)

        points = (torch.tensor(input_points_), torch.tensor(input_labels_))        
        prompt_feature = self.prompt_encoder(points=points, boxes=None, masks=None)
        image_pe=self.prompt_encoder.get_dense_pe()    
 
        low_res_masks, iou_pred = self.model.mask_decoder(
                image_embeddings = x["imgenc"],
                image_pe = image_pe,
                sparse_prompt_embeddings = prompt_feature[0],
                dense_prompt_embeddings = prompt_feature[1],
                multimask_output = False,
            )        
        #print("len low_res_masks", len(low_res_masks))
        #post process mask
        masks = F.interpolate(low_res_masks, (683, 1024), mode="bilinear",align_corners=False,)
        masks = masks[..., : 1024, : 1024]
        masks = F.interpolate(masks, (683, 1024), mode="bilinear", align_corners=False)

        #masks1 = self.postprocess_masks(
        #        masks= low_res_masks,
        #        input_size=input_img_size,
        #        original_size=org_img_size,
        #    )
        mask_threshold = 0.0
        masks = masks >  mask_threshold
        print("len mask1", len(masks), masks.shape)

        #post process result
        processed_results = []

        boxes = Boxes(torch.tensor([[260.1769, 168.1712, 701.4138, 401.5516]]))
        scores = torch.tensor([0.9830])
        classes = torch.tensor([48])
        masks = torch.rand(1, 683, 1024)  # Example binary mask

        # Create an instance
        instances = Instances(image_size=(683, 1024))
        instances['pred_boxes'] = boxes
        instances['scores'] = scores
        instances['pred_classes'] = classes
        instances['pred_masks'] = masks  # ✅ Now a real tensor

        # Wrap in result
        result = [f"{{'instances': {instances}}}"] 
        print("result", result)

        processed_results.append({"instances": result})

        return  result #processed_results


    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        # test
        enc = self._input_to_image_encoder(self, x)
        dec = self._image_encoder_to_output(enc)        

        return dec

    #@property
    #def cfg(self):
    #    return self._cfg


@register_vision_model("sam_vit_h_4b8939")
class sam_vit_h_4b8939(SAM):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("sam_vit_b_01ec64")
class sam_vit_b_01ec64(SAM):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("sam_vit_l_0b3195")
class sam_vit_l_0b3195(SAM):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)
