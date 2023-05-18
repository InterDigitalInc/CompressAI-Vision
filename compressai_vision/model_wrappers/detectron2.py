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

import torch
from .base_wrapper import BaseWrapper

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

class Rcnn_X_101_FPN(BaseWrapper):
    def __init__(self, running_on='cpu', **kwargs):
        super().__init__(running_on)

        cfg = get_cfg()
        #cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
        #self.model = 
        # self.aug = aug
        self.backbone = model.backbone
        self.proposal_generator = model.proposal_generator
        self.roi_heads = model.roi_heads
        self.postprocess = model._postprocess

        assert self.proposal_generator is not None

    def preInputTensor(self, img, img_id):
        """

        :param img: numpy BGR image (h,w,3)

        """
        height, width = img.shape[:2]
        if self.aug is not None:
            image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {
            "image": image,
            "height": height,
            "width": width,
            "image_id": img_id,
        }
        return [
            inputs,
        ]

    def fromInput2R2(self, x):
        # [batch, channel, height, width]
        # Resnet FPN
        stem_out = self.backbone.bottom_up.stem(x)
        r2_out = self.backbone.bottom_up.res2(stem_out)
        return r2_out

    def fromR22FPNFeatures(self, x):
        lateral_convs = [
            self.backbone.fpn_lateral5,
            self.backbone.fpn_lateral4,
            self.backbone.fpn_lateral3,
            self.backbone.fpn_lateral2,
        ]
        output_convs = [
            self.backbone.fpn_output5,
            self.backbone.fpn_output4,
            self.backbone.fpn_output3,
            self.backbone.fpn_output2,
        ]

        # Resnet FPN
        r2_out = x
        r3_out = self.backbone.bottom_up.res3(r2_out)
        r4_out = self.backbone.bottom_up.res4(r3_out)
        r5_out = self.backbone.bottom_up.res5(r4_out)

        bottom_up_features = {
            "res2": r2_out,
            "res3": r3_out,
            "res4": r4_out,
            "res5": r5_out,
        }
        # End

        results = []
        prev_features = lateral_convs[0](
            bottom_up_features[self.backbone.in_features[-1]]
        )
        results.append(output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for features, lateral_conv, output_conv in zip(
            self.backbone.in_features[-2::-1],
            lateral_convs[1:],
            output_convs[1:],
        ):
            features = bottom_up_features[features]
            top_down_features = F.interpolate(
                prev_features, scale_factor=2.0, mode="nearest"
            )

            # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
            lateral_features = lateral_conv.forward(features)
            prev_features = lateral_features + top_down_features
            if self.backbone._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv.forward(prev_features))

        if self.backbone.top_block is not None:
            if self.backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[
                    self.backbone.top_block.in_feature
                ]
            else:
                top_block_in_feature = results[
                    self.backbone._out_features.index(
                        self.backbone.top_block.in_feature
                    )
                ]

            results.extend(self.backbone.top_block(top_block_in_feature))

        assert len(self.backbone._out_features) == len(results)

        features = {f: res for f, res in zip(self.backbone._out_features, results)}

        return features

    def inferenceFromFPN(self, inputs, features):
        """
            Run inference on input features.

        inputs:
        - iput images are required
        - features

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """

        images = self.model.preprocess_image(inputs)

        proposals, _ = self.proposal_generator(images, features, None)

        results, _ = self.roi_heads(images, features, proposals, None)

        assert (
            not torch.jit.is_scripting()
        ), "Scripting is not supported for postprocess."
        return self.model._postprocess(results, inputs, images.image_sizes)

    def InputSize(self, inputs):
        return self.model.preprocess_image(inputs).tensor.size()

    @torch.no_grad()
    def forward(self, inputs):
        """

        :param img: numpy BGR image (h,w,3)

        """
        # whether the model expects BGR inputs or RGB
        image = self.model.preprocess_image(inputs)
        r2 = self.fromInput2R2(image.tensor)
        features = self.fromR22FPNFeatures(r2)

        # compress
        # call feature compression module here

        # infer heads
        results = self.inferenceFromFPN(features)

        return results

    def from_input_to_features(self, inputs):
        image = self.model.preprocess_image(inputs)
        r2 = self.fromInput2R2(image.tensor)
        features = self.fromR22FPNFeatures(r2)
        return features



#class Faster_Rcnn_X_101_32x8d_FPN_3x()
