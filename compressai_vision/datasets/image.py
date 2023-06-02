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


import base64
import json
from pathlib import Path
from typing import Dict, List

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.samplers import InferenceSampler
from PIL import Image
from torch.utils.data import Dataset

from compressai_vision import register_dataset
from compressai_vision.utils import logger


def deccode_compressed_rle(data):
    assert isinstance(data, Dict) or isinstance(data, List)

    if isinstance(data, Dict):
        data = list(data.values())

    for anno in data:
        segm = anno.get("segmentation", None)
        if segm:
            # Decode compressed RLEs with base64 to be compatible with pycoco tools
            if type(segm) != list and type(segm["counts"]) != list:
                segm["counts"] = base64.b64decode(segm["counts"])


class BaseDataset(Dataset):
    def __init__(self, dataset_name):
        self.sampler = None
        self.collate_fn = None
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.dataset_name


class BaseDatasetToFeedDetectron2(MapDataset):
    def __init__(self, dataset_name, dataset, cfg, images_folder, annotation_path):
        self.dataset_name = dataset_name

        self.annotation_path = annotation_path
        self.images_folder = images_folder

        try:
            DatasetCatalog.get(dataset_name)
        except KeyError:
            logger.warning(
                __name__,
                f'It looks a new dataset. The new dataset "{dataset_name}" is successfully registred in DataCatalog now.',
            )
            register_coco_instances(
                dataset_name, {}, self.annotation_path, self.images_folder
            )

        self.sampler = InferenceSampler(len(dataset))

        def bypass_collator(batch):
            return batch

        self.collate_fn = bypass_collator

        dataset = DatasetFromList(dataset, copy=False)
        mapper = DatasetMapper(cfg, False)

        super().__init__(dataset, mapper)

    def get_dataset_name(self):
        return self.dataset_name

    def get_annotation_path(self):
        return self.annotation_path

    def get_images_folder(self):
        return self.images_folder


# @register_dataset("ImageFolder")
class ImageFolder(BaseDataset):
    """Load an image folder database. testing image samples
    are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block::
        - rootdir/
            - img000.png
            - img001.png
    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        use_BGR (Bool): if True the color order of the sample is BGR otherwise RGB returned
    """

    def __init__(
        self, dataset_name, root, transform=None, ret_name=False, use_BGR=False
    ):
        super().__init__(dataset_name)

        splitdir = Path(root)

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in sorted(splitdir.iterdir()) if f.is_file()]

        self.use_BGR = use_BGR
        self.transform = transform
        self.ret_name = ret_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")

        if self.use_BGR is True:
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))

        if self.transform:
            if self.ret_name is True:
                return (self.transform(img), str(self.samples[index]))
            return self.transform(img)

        if self.ret_name is True:
            return (img, str(self.samples[index]))

        return img

    def __len__(self):
        return len(self.samples)


@register_dataset("MPEGOIV6")
class MPEGOIV6_ImageFolder(BaseDatasetToFeedDetectron2):
    """Load an image folder database to support testing image samples from MPEG-OpenimagesV6:

    .. code-block::
        - mpeg-oiv6/
            - annoations/
                -
            - images/
                - 452c856678a9b284.jpg
                - ....jpg

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        use_BGR (Bool): if True the color order of the sample is BGR otherwise RGB returned
    """

    def __init__(
        self,
        dsroot,
        cfg,
        img_folder_name=None,
        annotation_file_name="mpeg-oiv6-segmentation-coco.json",
        dataset_name="mpeg-oiv6-segmentation",
    ):
        #                    annotation_file_name="mpeg-oiv6-detection-coco.json",
        #                    dataset_name="mpeg-oiv6-detection"):

        if img_folder_name is None:
            images_dir = Path(dsroot) / "images"
        else:
            images_dir = Path(dsroot) / img_folder_name

        self.task = "detection"
        if "segmentation" in dataset_name:
            self.task = "segmentation"

        annotations_file = Path(dsroot) / "annotations" / annotation_file_name

        if not images_dir.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{images_dir}"')

        if not annotations_file.is_file():
            raise RuntimeError(f'Invalid annotation file "{annotations_file}"')

        dataset = load_coco_json(
            annotations_file, images_dir, dataset_name=dataset_name
        )

        # if self.task == 'segmentation':
        #    self.deccode_compressed_rle(dataset)

        super().__init__(dataset_name, dataset, cfg, images_dir, annotations_file)

    def get_min_max_across_tensors(self):
        if self.task == "segmentation":
            maxv = 28.397489547729492
            minv = -26.426830291748047
            return (minv, maxv)

        assert self.task == "detection"
        maxv = 20.246625900268555
        minv = -23.09193229675293
        return (minv, maxv)


@register_dataset("SFUHW")
class SFUHW_ImageFolder(BaseDatasetToFeedDetectron2):
    """Load an image folder database with Detectron2 Cfg. testing image samples
    and annotations are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block::
        - rootdir/
            - images
                - img000.png
                - img001.png
                - imgxxx.png
            - annotations
                - xxxx.json
    Args:
        root (string): root directory of the dataset

    """

    def __init__(
        self,
        dsroot,
        cfg,
        img_folder_name=None,
        annotation_file_name=None,
        dataset_name="sfu-hw-object-v1",
    ):
        if img_folder_name is None:
            images_dir = Path(dsroot) / "images"
        else:
            images_dir = Path(dsroot) / img_folder_name

        if annotation_file_name is None:
            annotations_dir = Path(dsroot) / "annotations"

            if not annotations_dir.is_dir():
                raise RuntimeError(f'Invalid annotation directory "{annotations_dir}"')

            annotations = [
                f
                for f in sorted(annotations_dir.iterdir())
                if f.is_file() and f.suffix[1:] == "json"
            ]

            if len(annotations) != 1:
                raise RuntimeError(
                    f"The number of json file for the samples annotations must be 1, but got {len(annotations)}"
                )
            annotations_file = annotations[0]

        else:
            annotations_file = Path(dsroot) / "annotations" / annotation_file_name

        if not images_dir.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{images_dir}"')

        if not annotations_file.is_file():
            raise RuntimeError(f'Invalid annotation file "{annotations_file}"')

        dataset = load_coco_json(
            annotations_file, images_dir, dataset_name=dataset_name
        )

        super().__init__(dataset_name, dataset, cfg, images_dir, annotations_file)

    def get_min_max_across_tensors(self):
        # from mpeg-fcvcm
        minv = -17.884761810302734
        maxv = 16.694171905517578
        return (minv, maxv)


@register_dataset("COCO")
class COCO_ImageFolder(BaseDatasetToFeedDetectron2):
    """Load an image folder database with Detectron2 Cfg. testing image samples
    and annotations are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block::
        - rootdir/
            - [train_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg
            - [validation_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg
            - [test_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg
            - annotations
                - [instances_val].json
                - [captions_val].json
                - ...
    Args:
        root (string): root directory of the dataset

    """

    def __init__(
        self,
        dsroot,
        cfg,
        img_folder_name="val2017",
        annotation_file_name="instances_val2017.json",
        dataset_name="mpeg-coco",
    ):
        images_dir = Path(dsroot) / img_folder_name
        annotations_file = Path(dsroot) / "annotations" / annotation_file_name

        if not images_dir.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{images_dir}"')

        if not annotations_file.is_file():
            raise RuntimeError(f'Invalid annotation file "{annotations_file}"')

        dataset = load_coco_json(
            annotations_file, images_dir, dataset_name=dataset_name
        )

        super().__init__(dataset_name, dataset, cfg, images_dir, annotations_file)
