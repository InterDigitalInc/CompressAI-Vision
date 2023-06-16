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
from pathlib import Path
from typing import Dict, List

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.samplers import InferenceSampler
from PIL import Image
from torch.utils.data import Dataset

from compressai_vision.registry import register_datacatalog, register_dataset
from compressai_vision.utils import logger


def bypass_collator(batch):
    return batch


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


@register_dataset("DefaultDataset")
class DefaultDataset(Dataset):
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
        self,
        root,
        dataset_name,
        imgs_folder: str = "valid",
        **kwargs,
    ):
        super().__init__()

        self.sampler = None
        self.collate_fn = None

        splitdir = Path(root) / imgs_folder

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in sorted(splitdir.iterdir()) if f.is_file()]

        self.use_BGR = kwargs["use_BGR"]
        self.transform = kwargs["transforms"]
        self.ret_name = kwargs["ret_name"]
        self.dataset_name = dataset_name

        self.mapDataset = None
        if "cfg" in kwargs:
            if kwargs["cfg"] is not None:
                self.sampler = InferenceSampler(len(kwargs["dataset"]))
                self.collate_fn = bypass_collator

                _dataset = DatasetFromList(kwargs["dataset"].dataset, copy=False)
                mapper = DatasetMapper(kwargs["cfg"], False)

                self.mapDataset = MapDataset(_dataset, mapper)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        if self.mapDataset:
            return self.mapDataset[index]

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
        if self.mapDataset:
            return len(self.mapDataset)

        return len(self.samples)


@register_dataset("Detectron2Dataset")
class Detectron2Dataset(Dataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__()

        self.dataset_name = dataset_name

        self.annotation_path = Path(root) / "annotations" / kwargs["annotation_file"]
        self.images_folder = Path(root) / imgs_folder

        assert (
            self.annotation_path == kwargs["dataset"].annotation_path
            and self.images_folder == kwargs["dataset"].imgs_folder_path
        )

        self.dataset = kwargs["dataset"].dataset

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

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(self.dataset, copy=False)
        mapper = DatasetMapper(kwargs["cfg"], False)

        self.mapDataset = MapDataset(_dataset, mapper)

    def __getitem__(self, idx):
        return self.mapDataset[idx]

    def __len__(self):
        return len(self.mapDataset)


class DataCatalog:
    def __init__(
        self,
        root,
        imgs_folder="images",
        annotation_file="sample.json",
        dataset_name="sample_dataset",
    ):
        _imgs_folder = Path(root) / imgs_folder
        if not _imgs_folder.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{_imgs_folder}"')

        _annotation_file = Path(root) / "annotations" / annotation_file
        if not _annotation_file.is_file():
            raise RuntimeError(f'Invalid annotation file "{_annotation_file}"')

        self._dataset_name = dataset_name
        self._dataset = None
        self._imgs_folder = _imgs_folder
        self._annotation_file = _annotation_file

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset(self):
        return self._dataset

    @property
    def annotation_path(self):
        return self._annotation_file

    @property
    def imgs_folder_path(self):
        return self._imgs_folder

    def __len__(self):
        return len(self._dataset)

        # super().__init__(dataset_name, dataset, cfg, imgs_folder_path, annotations_file)


@register_datacatalog("MPEGOIV6")
class MPEGOIV6(DataCatalog):
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
        root,
        imgs_folder="images",
        annotation_file="mpeg-oiv6-segmentation-coco.json",
        dataset_name="mpeg-oiv6-segmentation",
        # annotation_file_name="mpeg-oiv6-detection-coco.json",
        # dataset_name="mpeg-oiv6-detection",
    ):
        super().__init__(root, imgs_folder, annotation_file, dataset_name)

        self._dataset = load_coco_json(
            self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
        )
        self.task = "detection"
        if "segmentation" in dataset_name:
            self.task = "segmentation"

        # TODO [hyomin]
        # if self.task == 'segmentation':
        #    self.deccode_compressed_rle(dataset)

    def get_min_max_across_tensors(self):
        if self.task == "segmentation":
            maxv = 28.397489547729492
            minv = -26.426830291748047
            return (minv, maxv)

        assert self.task == "detection"
        maxv = 20.246625900268555
        minv = -23.09193229675293
        return (minv, maxv)


@register_datacatalog("SFUHW")
class SFUHW(DataCatalog):
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
        root,
        imgs_folder="images",
        annotation_file=None,
        dataset_name="sfu-hw-object-v1",
    ):
        super().__init__(root, imgs_folder, annotation_file, dataset_name)

        self._dataset = load_coco_json(
            self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
        )

    def get_min_max_across_tensors(self):
        # from mpeg-fcvcm
        minv = -17.884761810302734
        maxv = 16.694171905517578
        return (minv, maxv)


@register_datacatalog("COCO")
class COCO(DataCatalog):
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
        root,
        imgs_folder="val2017",
        annotation_file="instances_val2017.json",
        dataset_name="mpeg-coco",
    ):
        super().__init__(root, imgs_folder, annotation_file, dataset_name)

        self._dataset = load_coco_json(
            self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
        )

    def get_min_max_across_tensors(self):
        raise NotImplementedError
