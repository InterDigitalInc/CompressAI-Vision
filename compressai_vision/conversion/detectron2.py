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

"""From 51 dataset into Detectron2-compatible dataset
"""
from math import floor

# import cv2
# import detectron2
import torch

# from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

# from fiftyone import ProgressBar
from fiftyone.core.dataset import Dataset
from fiftyone.core.labels import Detection, Detections
from PIL import Image


def findLabels(dataset: Dataset, detection_field: str = "detections") -> list:
    return dataset.distinct("%s.detections.label" % detection_field)


class FO2DetectronDataset(torch.utils.data.Dataset):
    """A class to construct a Detectron2 dataset from a FiftyOne dataset.
    Subclass of ``torch.utils.data.Dataset``.

    :param fo_dataset: fiftyone dataset
    :param detection_field: name of member in the FiftyOne Sample where the
        detector (ground truth) is put into.  Default: "detections".
    :param model_catids: a list of category labels as provided from Detectron2
        model's metadata.  Used to transform fiftyone category label into an
        index number used by Detectron2

    NOTE: Usually we are more interested in going from Detectron results to
    FiftyOne format, so you might not use this torch Dataset class that much

    refs:

    - https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html
    - https://towardsdatascience.com/stop-wasting-time-with-pytorch-datasets-17cac2c22fa8
    - https://medium.com/voxel51/how-to-train-your-dragon-detector-a35ed4672ca7


    WARNING: at the moment, only detection (not segmentation) is supported
    """

    def __init__(
        self,
        fo_dataset: Dataset = None,
        detection_field="detections",
        # let's use "detections" or "ground-truths" for GT and "predictions" for
        # detectron2-give predictions
        model_catids=[],  # noqa: B006
        # TODO (sampsa) "Do not use mutable data structures for argument defaults"
    ):
        assert fo_dataset is not None, "please provide fo_dataset (fiftyone dataset)"
        assert (
            len(model_catids) > 0
        ), "please provide MODEL's ORIGINAL category label list.  Get his from detectron2 model's metadata."
        self.fo_dataset = fo_dataset
        self.detection_field = detection_field
        self.model_catids = model_catids
        self.img_paths = self.fo_dataset.values(
            "filepath"
        )  # list of all filepaths in the dataset
        # Get list of distinct labels that exist in the view
        # self.classes = self.fo_dataset.distinct(
        #    "%s.detections.label" % detection_field
        # )

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.fo_dataset[
            img_path
        ]  # datasets are allowed to be index with ids, filenames, etc. only (not with plain integers)
        # metadata = sample.metadata
        img = Image.open(img_path)

        """Example detectron2 formatted sample:

        ::

            {'file_name': '/home/sampsa/fiftyone/openimagev6_mpeg_vcm_small_COCO/data/001997021f01f208.jpg',
            'height': 1024,
            'width': 760,
            'image_id': 1,
            'annotations': [
                {'iscrowd': 0,
                'bbox': [219.874232, 250.02800128, 58.95175599999998, 74.84913663999998],
                'category_id': 32,
                'bbox_mode': <BoxMode.XYWH_ABS: 1>},
                {'iscrowd': 0,
                'bbox': [430.188652, 410.87401984, 82.85117200000002, 95.55209215999997],
                'category_id': 32,
                'bbox_mode': <BoxMode.XYWH_ABS: 1>}
            ]


        Example fiftyone sample:

        ::

            <Sample: {
                'id': '62e55b386fa654ded87ece8e',
                'media_type': 'image',
                'filepath': '/tmp/kikkelis/data/83bf8172abed42d7.jpg',
                'tags': BaseList([]),
                'metadata': None,
                'open_images_id': '83bf8172abed42d7',
                'detections':   # can have several members whose class is Detections.  Typically named "ground_truth" or "prediction"
                    # evaluators then use both fields
                    <Detections: {
                    'detections': BaseList([
                        <Detection: {
                            'id': '62e55b386fa654ded87ecda5',
                            'attributes': BaseDict({}),
                            'tags': BaseList([]),
                            'label': 'bird',
                            'bounding_box': BaseList([
                                0.27138644,
                                0.086283185,
                                0.72861356,
                                0.913716815,
                            ]),
                            'mask': None,
                            'confidence': None,
                            'index': None,
                            'IsOccluded': False,
                            'IsTruncated': False,
                            'IsGroupOf': False,
                            'IsDepiction': False,
                            'IsInside': False,
                        }>,
                    ]),
                }>,
            }>

        OpenImageV6 bbox format: xMin, xMax, yMin, yMax (or starting with yMin, depending what tool you use(!)
        https://stackoverflow.com/questions/55832578/how-to-make-sense-of-open-images-datasets-bounding-box-annotations)
        fiftyone bbox format: all relative coordinates: [x0, y0, w, h] (origo at left up)
        Detectron2 bbox format: see: https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode
        (OpenImageV6 bbox seems to be equal to BoxMode.XYWH_REL = 3)

        Detectron2 visualizer tool says for BoxMode.XYWH_REAL:

        ::

            AssertionError: Relative mode not yet supported!

        So I deduce that all other than BoxMode.XYWH_ABS are in reality, useless

        """

        d = {
            "file_name": sample.filepath,
            "height": img.height,
            "width": img.width,
            "image_id": sample.id,
        }

        annotations = []
        # detections = sample.detections.detections # so could that last one be detections, ground_truths, etc.?
        # print(sample)
        detections = None
        if sample[self.detection_field] is not None:
            detections = sample[self.detection_field].detections
        if detections is not None:
            for detection in detections:
                x_, y_, w_, h_ = detection.bounding_box
                op = floor
                bbox = [
                    op(x_ * img.width),
                    op(y_ * img.height),
                    op(w_ * img.width),
                    op(h_ * img.height),
                ]
                try:
                    n = self.model_catids.index(detection.label)
                except ValueError:
                    print(
                        "found a label name that is not in the 'model_catids' provided"
                    )
                    raise
                annotations.append(
                    {
                        "iscrowd": 0,
                        "bbox": bbox,
                        # category_id = 0, # from label to catid
                        "category_id": n,
                        # "bbox_mode" : BoxMode.XYWH_REL # does not work
                        "bbox_mode": BoxMode.XYWH_ABS,
                    }
                )

        d["annotations"] = annotations
        return d

    def __len__(self):
        return len(self.img_paths)


def detectron251(
    res,
    model_catids: list = [],  # noqa: B006
    # TODO (sampsa) "Do not use mutable data structures for argument defaults"
    allowed_labels: list = None,
    verbose=False,
) -> list:
    """Detectron2 formatted results, i.e. ``{'instances': Instances}`` into FiftyOne-formatted results

    This works for detectors and instance segmentation, where a segmentation is always accompanied with a bounding box

    :param res: Detectron2 predictor output (a dictionary ``{'instances': Instances}``)
    :param model_catids: A category label list, as provided by Detectron2 model's metadata

    Returns FiftyOne ``Detections`` instance that can be attached to a FiftyOne ``Sample`` instance.
    """
    assert (
        len(model_catids) > 0
    ), "please provide MODEL's ORIGINAL category label list.  Get it from detectron2 model's metadata."
    """

    Which you would do with:

    ::

        model_dataset=cfg.DATASETS.TRAIN[0]
        model_meta=MetadataCatalog.get(model_dataset)
        model_meta.thing_classes

    """

    instances = res["instances"]

    """
    For example:

    ::

        Instances(num_instances=5, image_height=447, image_width=1024, fields=[pred_boxes: Boxes(tensor([[130.7466,  12.2867, 962.2325, 354.4287],
            [941.1395, 268.9208, 978.8932, 300.9435],
            [891.1142, 275.1706, 942.4617, 299.9345],
            [112.7023, 152.5475, 307.6531, 241.2931],
            [815.8085, 311.5709, 898.5360, 346.5404]])),
            scores: tensor([0.9964, 0.9494, 0.9204, 0.7443, 0.6773]), pred_classes: tensor([4, 7, 7, 4, 7])])

    """
    dets = []
    # all models give scores & pred_classes
    # for bbox, score, pred_class in zip(instances.pred_boxes, instances.scores, instances.pred_classes):
    for i, score in enumerate(instances.scores):
        bbox = None
        mask = None
        class_index = instances[i].pred_classes.detach().item()
        # index to label
        try:
            label = model_catids[class_index]
        except IndexError:
            print(
                "model gave pred_class",
                class_index,
                "but the model_catids provided length is only",
                len(model_catids),
            )
            raise
        if allowed_labels is None:
            pass
        elif label not in allowed_labels:
            if verbose:
                print("detectron251: skipping label", label)
            continue
        # print(bbox.to("cpu").tolist(), score.to("cpu").item())
        height, width = instances.image_size

        """https://voxel51.com/docs/fiftyone/api/fiftyone.core.labels.html#fiftyone.core.labels.Detection
        bbox format: relative (0->1) bbox coordinates: [<top-left-x>, <top-left-y>, <width>, <height>]
        mask: an instance segmentation mask for the detection within its bounding box, which should be a 2D binary or 0/1 integer numpy array
        """
        # bboxes
        if hasattr(instances, "pred_boxes"):
            boxObject = instances.pred_boxes[i]  # indexing returns a Boxes object
            for t in boxObject:  # noqa: B007
                # TODO (sampsa) "Loop control variable 't' not used within the loop body"
                # so annoying.. only way to get the tensor is to iterate
                pass
            x, y, x2, y2 = t.detach().tolist()  # detectron2: abs coordinates
            bbox = [
                x / width,
                y / height,
                (x2 - x) / width,
                (y2 - y) / height,
            ]  # fiftyone: rex, rely, relw, relh
            # print(bbox)
        # segmentation
        if hasattr(instances, "pred_masks"):
            mask = instances.pred_masks[i].detach().numpy()
            x_ = floor(x)
            y_ = floor(y)
            x2_ = floor(x2)
            y2_ = floor(y2)
            small_mask = mask[y_:y2_, x_:x2_]

        if bbox is not None:
            if mask is None:
                dets.append(
                    Detection(
                        label=label, confidence=score.detach().item(), bounding_box=bbox
                    )
                )
            else:  # we have also a mask
                dets.append(
                    Detection(
                        label=label,
                        confidence=score.detach().item(),
                        bounding_box=bbox,
                        mask=small_mask,
                    )
                )

    detections = Detections(detections=dets)
    return detections
