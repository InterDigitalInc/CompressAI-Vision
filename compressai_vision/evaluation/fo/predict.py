# Copyright (c) 2022, InterDigital Communications, Inc
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

import math
import os

import cv2

from detectron2.data import MetadataCatalog
from fiftyone import ProgressBar
from fiftyone.core.dataset import Dataset

from compressai_vision.conversion.detectron2 import detectron251, findLabels
from compressai_vision.evaluation.pipeline.base import EncoderDecoder


def annexPredictions(
    predictor=None,
    fo_dataset: Dataset = None,
    detection_field: str = "detections",
    predictor_field: str = "detectron-predictions",
    encoder_decoder=None,  # compressai_vision.evaluation.pipeline.base.EncoderDecoder
):
    """Run detector on a dataset and annex predictions of the detector into the dataset.

    :param predictor: A Detectron2 predictor
    :param fo_dataset: Fiftyone dataset
    :param detection_field: Which dataset member to use for ground truths.  Default:    "detections"
    :param predictor_field: Which dataset member to use for saving the Detectron2 results.  Default: "detectron-predictions"
    :param encoder_decoder: (optional) a ``compressai_vision.evaluation.pipeline.EncoderDecoder`` subclass instance to apply on the image before detection
    """
    assert predictor is not None, "provide Detectron2 predictor"
    assert fo_dataset is not None, "provide fiftyone dataset"
    if encoder_decoder is not None:
        assert issubclass(
            encoder_decoder.__class__, EncoderDecoder
        ), "encoder_decoder instances needs to be a subclass of EncoderDecoder"

    model_meta = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])

    """we don't need this!
    d2_dataset = FO2DetectronDataset(
        fo_dataset=fo_dataset,
        detection_field=detection_field,
        model_catids = model_meta.things_classes,
        )
    """
    try:
        _ = findLabels(fo_dataset, detection_field=detection_field)
    except ValueError:
        print(
            "your ground truths suck: samples have no member",
            detection_field,
            "will set allowed_labels to empty list",
        )
        # allowed_labels = []

    # add predicted bboxes to each fiftyone Sample
    bpp_sum = 0
    with ProgressBar(fo_dataset) as pb:
        for sample in fo_dataset:
            # sample.filepath
            path = sample.filepath
            im = cv2.imread(path)
            tag = path.split(os.path.sep)[-1].split(".")[
                0
            ]  # i.e.: /path/to/some.jpg --> some.jpg --> some
            # print(tag)
            if encoder_decoder is not None:
                # before using detector, crunch through
                # encoder/decoder
                bpp, im = encoder_decoder.BGR(
                    im, tag=tag
                )  # include a tag for cases where EncoderDecoder uses caching
                bpp_sum += bpp
            res = predictor(im)
            predictions = detectron251(
                res,
                model_catids=model_meta.thing_classes,
                # allowed_labels=allowed_labels # not needed, really
            )  # fiftyone Detections object
            sample[predictor_field] = predictions
            # we could attach the bitrate to each detection, of course
            # if encoder_decoder is not None:
            #    sample["bpp"]=bpp
            sample.save()
            pb.update()
    if encoder_decoder is None:
        return None
    elif len(fo_dataset) > 0:
        return bpp_sum / len(fo_dataset)
    else:
        return math.nan
