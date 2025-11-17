# Copyright (c) 2022-2024, InterDigital Communications, Inc
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
import logging
import re

from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.samplers import InferenceSampler
from detectron2.data.transforms import AugmentationList
from detectron2.utils.serialize import PicklableWrapper
from PIL import Image
from torch.utils.data import Dataset

from compressai_vision.registry import register_datacatalog, register_dataset

from .utils import (
    JDECustomMapper,
    LinearMapper,
    MMPOSECustomMapper,
    SAMCustomMapper,
    YOLOXCustomMapper,
)


def manual_load_data(path, ext):
    img_list = sorted(glob(f"{path}/*.{ext}"))

    datalist = []

    for img_addr in img_list:
        img_id = Path(img_addr).stem
        img = Image.open(img_addr)
        fW, fH = img.size

        d = {
            "file_name": img_addr,
            "height": fH,
            "width": fW,
            "image_id": img_id,
            "annotations": None,
        }

        datalist.append(d)

    return datalist


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
            if type(segm) is not list and type(segm["counts"]) is not list:
                segm["counts"] = base64.b64decode(segm["counts"])


class BaseDataset(Dataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_name = dataset_name

        self.annotation_path = None
        if "annotation_file" in kwargs:
            if kwargs["annotation_file"].lower() != "none":
                self.annotation_path = Path(root) / kwargs["annotation_file"]
                assert self.annotation_path == kwargs["dataset"].annotation_path

        self.seqinfo_path = None
        if "seqinfo" in kwargs:
            if kwargs["seqinfo"].lower() != "none":
                self.seqinfo_path = kwargs["dataset"].seqinfo_path

        self.input_agumentation_bypass = False
        if "input_augmentation_bypass" in kwargs:
            self.input_agumentation_bypass = kwargs["input_augmentation_bypass"]
            if self.input_agumentation_bypass:
                self.logger.warning(
                    "The vision model may or may not support the feature of input agumentation bypass\n"
                )

        self.images_folder = Path(root) / imgs_folder
        assert self.images_folder == kwargs["dataset"].imgs_folder_path

        self.sampler = None
        self.collate_fn = None
        self.mapDataset = None
        self.org_mapper_func = None

        self.thing_classes = []
        self.thing_dataset_id_to_contiguous_id = []


@register_dataset("DefaultDataset")
class DefaultDataset(BaseDataset):
    """
    Loads an image folder database. testing image samples
    are respectively stored in separate directories
    (Currently, this class does not support any of the training related operations):

    .. code-block:: text

        |--rootdir
          |-- img000.png
          |-- img001.png

    Attributes
    ----------
        root : string
            root directory of the dataset
        transform : (callable, optional)
            a function or transform that takes in a PIL image and returns a transformed version
        use_BGR : Bool
            if True the color order of the sample is BGR otherwise RGB returned
    """

    def __init__(
        self,
        root,
        dataset_name,
        imgs_folder: str = "valid",
        **kwargs,
    ):
        super().__init__(root, dataset_name, imgs_folder, **kwargs)

        if not self.images_folder.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in sorted(self.images_folder.iterdir()) if f.is_file()]

        self.use_BGR = kwargs["use_BGR"]
        self.transform = kwargs["transforms"]
        self.ret_name = kwargs["ret_name"]

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(kwargs["dataset"].dataset, copy=False)

        if "cfg" in kwargs:
            if kwargs["cfg"] is not None:
                mapper = DatasetMapper(kwargs["cfg"], False)

                self.mapDataset = MapDataset(_dataset, mapper)

                return

        self.mapDataset = MapDataset(_dataset, LinearMapper(bgr=self.use_BGR))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        if self.mapDataset:
            return self.mapDataset[index]

        raise NotImplementedError
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
class Detectron2Dataset(BaseDataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__(root, dataset_name, imgs_folder, **kwargs)

        self.dataset = kwargs["dataset"].dataset

        try:
            DatasetCatalog.get(dataset_name)
        except KeyError:
            if self.annotation_path:
                register_coco_instances(
                    dataset_name, {}, self.annotation_path, self.images_folder
                )
                self.logger.info(f'"{dataset_name}" successfully registred.')

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(self.dataset, copy=False)

        if kwargs["linear_mapper"] is True:
            mapper = LinearMapper()
        else:
            assert (
                kwargs["cfg"] is not None
            ), "A proper mapper information via cfg must be provided"
            mapper = DatasetMapper(kwargs["cfg"], False)

        self._org_mapper_func = PicklableWrapper(DatasetMapper(kwargs["cfg"], False))

        if self.input_agumentation_bypass:
            emptyAugList = AugmentationList([])
            mapper.augmentations = emptyAugList

            if hasattr(self._org_mapper_func, "_obj"):
                self._org_mapper_func._obj.augmentations = emptyAugList
            else:
                self._org_mapper_func.augmentations = emptyAugList

        self.mapDataset = MapDataset(_dataset, mapper)

        metaData = MetadataCatalog.get(dataset_name)
        try:
            self.thing_classes = metaData.thing_classes
            self.thing_dataset_id_to_contiguous_id = (
                metaData.thing_dataset_id_to_contiguous_id
            )
        except AttributeError:
            self.logger.warning("No attribute: thing_classes")

    def get_org_mapper_func(self):
        return self._org_mapper_func

    def __getitem__(self, idx):
        return self.mapDataset[idx]

    def __len__(self):
        return len(self.mapDataset)


@register_dataset("SamDataset")
class SamDataset(BaseDataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__(root, dataset_name, imgs_folder, **kwargs)

        self.dataset = kwargs["dataset"].dataset

        try:
            DatasetCatalog.get(dataset_name)
        except KeyError:
            if self.annotation_path:
                register_coco_instances(
                    dataset_name, {}, self.annotation_path, self.images_folder
                )
                self.logger.info(f'"{dataset_name}" successfully registred.')

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(self.dataset, copy=False)
        mapper = SAMCustomMapper(
            augmentation_bypass=kwargs["input_augmentation_bypass"]
        )

        self.mapDataset = MapDataset(_dataset, mapper)
        self._org_mapper_func = PicklableWrapper(mapper)

        metaData = MetadataCatalog.get(dataset_name)
        try:
            self.thing_classes = metaData.thing_classes
            self.thing_dataset_id_to_contiguous_id = (
                metaData.thing_dataset_id_to_contiguous_id
            )
        except AttributeError:
            self.logger.warning("No attribute: thing_classes")

    def get_org_mapper_func(self):
        return self._org_mapper_func

    def __getitem__(self, idx):
        return self.mapDataset[idx]

    def __len__(self):
        return len(self.mapDataset)


@register_dataset("TrackingDataset")
class TrackingDataset(BaseDataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__(root, dataset_name, imgs_folder, **kwargs)

        self.dataset = kwargs["dataset"].dataset

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(self.dataset, copy=False)

        if kwargs["linear_mapper"] is True:
            mapper = LinearMapper()
        else:
            mapper = JDECustomMapper(kwargs["patch_size"])

        self.mapDataset = MapDataset(_dataset, mapper)
        self._org_mapper_func = PicklableWrapper(JDECustomMapper(kwargs["patch_size"]))

    def get_org_mapper_func(self):
        return self._org_mapper_func

    def __getitem__(self, idx):
        return self.mapDataset[idx]

    def __len__(self):
        return len(self.mapDataset)


@register_dataset("YOLOXDataset")
class YOLOXDataset(BaseDataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__(root, dataset_name, imgs_folder, **kwargs)

        self.dataset = kwargs["dataset"].dataset

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(self.dataset, copy=False)

        if kwargs["linear_mapper"] is True:
            mapper = LinearMapper()
        else:
            mapper = YOLOXCustomMapper(kwargs["patch_size"])

        self.input_size = kwargs["patch_size"]
        self.mapDataset = MapDataset(_dataset, mapper)
        self._org_mapper_func = PicklableWrapper(
            YOLOXCustomMapper(kwargs["patch_size"])
        )

        metaData = MetadataCatalog.get(dataset_name)
        try:
            self.thing_classes = metaData.thing_classes
            self.thing_dataset_id_to_contiguous_id = (
                metaData.thing_dataset_id_to_contiguous_id
            )
        except AttributeError:
            self.logger.warning("No attribute: thing_classes")

    def get_org_mapper_func(self):
        return self._org_mapper_func

    def __getitem__(self, idx):
        return self.mapDataset[idx]

    def __len__(self):
        return len(self.mapDataset)


@register_dataset("MMPOSEDataset")
class MMPOSEDataset(BaseDataset):
    def __init__(self, root, dataset_name, imgs_folder, **kwargs):
        super().__init__(root, dataset_name, imgs_folder, **kwargs)

        self.dataset = kwargs["dataset"].dataset

        self.sampler = InferenceSampler(len(kwargs["dataset"]))
        self.collate_fn = bypass_collator

        _dataset = DatasetFromList(self.dataset, copy=False)

        if kwargs["linear_mapper"] is True:
            mapper = LinearMapper()
        else:
            mapper = MMPOSECustomMapper(kwargs["patch_size"])

        self.input_size = kwargs["patch_size"]
        self.mapDataset = MapDataset(_dataset, mapper)
        self._org_mapper_func = PicklableWrapper(
            MMPOSECustomMapper(kwargs["patch_size"])
        )

        metaData = MetadataCatalog.get(dataset_name)
        try:
            self.thing_classes = metaData.thing_classes
            self.thing_dataset_id_to_contiguous_id = (
                metaData.thing_dataset_id_to_contiguous_id
            )
        except AttributeError:
            self.logger.warning("No attribute: thing_classes")

    def get_org_mapper_func(self):
        return self._org_mapper_func

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
        seqinfo="seqinfo.ini",
        dataset_name="sample_dataset",
        ext=".png",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        _imgs_folder = Path(root) / imgs_folder
        if not _imgs_folder.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{_imgs_folder}"')

        self._annotation_file = None
        if annotation_file.lower() != "none":
            _annotation_file = Path(root) / annotation_file
            if not _annotation_file.is_file():
                raise RuntimeError(f'Invalid annotation file "{_annotation_file}"')
            self._annotation_file = _annotation_file
        else:  # annotation_file is not available
            self.logger.warning(
                "No annotation found, there may be no evaluation output based on groundtruth\n"
            )

        self._sequence_info_file = None
        if seqinfo.lower() != "none":
            _sequence_info_file = Path(root) / seqinfo
            if not _annotation_file.is_file():
                self.logger.warning(
                    f"Sequence information does not exist at the given path {_sequence_info_file}"
                )
                self._sequence_info_file = None
            else:
                self._sequence_info_file = _sequence_info_file
        else:  # seqinfo is not available
            self.logger.warning("No sequence information provided\n")

        self._dataset_name = dataset_name
        self._dataset = None
        self._imgs_folder = _imgs_folder
        self._img_ext = ext

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
    def seqinfo_path(self):
        return self._sequence_info_file

    @property
    def imgs_folder_path(self):
        return self._imgs_folder

    def __len__(self):
        return len(self._dataset)

        # super().__init__(dataset_name, dataset, cfg, imgs_folder_path, annotations_file)


@register_datacatalog("MPEGTVDTRACKING")
class MPEGTVDTRACKING(DataCatalog):
    """Load an image folder database to support testing image samples extracted from MPEG-TVD Objects Tracking videos:

    .. code-block:: none
        - mpeg-TVD-Tracking/
            - annoations/
                -
            - images/
                - 00001.png
                - ....png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
    """

    def __init__(
        self,
        root,
        imgs_folder="images",
        annotation_file="gt.txt",
        seqinfo="seqinfo.ini",
        dataset_name="mpeg-tvd-tracking",
        ext="png",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )
        from jde.utils.io import read_results

        self.data_type = "mot"
        gt_frame_dict = read_results(
            str(self.annotation_path), self.data_type, is_gt=True
        )
        gt_ignore_frame_dict = read_results(
            str(self.annotation_path), self.data_type, is_ignore=True
        )

        img_lists = sorted(self.imgs_folder_path.glob(f"*.{ext}"))

        assert len(gt_frame_dict) == len(gt_ignore_frame_dict)

        self._dataset = []
        self._gt_labels = gt_frame_dict
        self._gt_ignore_labels = gt_ignore_frame_dict

        for file_name in img_lists:
            img_id = file_name.name.split(f".{ext}")[0]

            new_d = {
                "file_name": str(file_name),
                "image_id": img_id,
                "annotations": {
                    "gt": gt_frame_dict.get(int(img_id), []),
                    "gt_ignore": gt_ignore_frame_dict.get(int(img_id), []),
                },
            }

            self._dataset.append(new_d)

    def get_ground_truth_labels(self, id: int):
        return {
            "gt": self._gt_labels.get(id, []),
            "gt_ignore": self._gt_ignore_labels.get(id, []),
        }

    def get_min_max_across_tensors(self):
        maxv = 48.58344268798828
        minv = -4.722218990325928
        return (minv, maxv)


@register_datacatalog("MPEGHIEVE")
class MPEGHIEVE(MPEGTVDTRACKING):
    """Load an image folder database to support testing image samples extracted from MPEG-HiEve videos:

    .. code-block:: none
        - mpeg-HiEve/
            - annoations/
                -
            - images/
                - 00001.png
                - ....png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
    """

    def __init__(
        self,
        root,
        imgs_folder="images",
        annotation_file="gt.txt",
        seqinfo="seqinfo.ini",
        dataset_name="mpeg-hieve-tracking",
        ext="png",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )

    def get_min_max_across_tensors(self):
        maxv = 11.823183059692383
        minv = -1.0795124769210815
        return (minv, maxv)


@register_datacatalog("MPEGSAM")
class MPEGSAM(DataCatalog):
    def __init__(
        self,
        root,
        imgs_folder="images",
        annotation_file="mpeg-oiv6-segmentation-coco_fortest.json",
        seqinfo="",
        dataset_name="mpeg-oiv6-sam",
        ext="",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )

        if self.annotation_path:
            self._dataset = load_coco_json(
                self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
            )
        else:
            self._dataset = manual_load_data(self.imgs_folder_path, "jpg")

        self.task = "sam"

    def get_min_max_across_tensors(self):
        if self.task == "sam":
            maxv = 28.397489547729492  # TODO these value are used for segmentation
            minv = -26.426830291748047
            return (minv, maxv)


@register_datacatalog("MPEGOIV6")
class MPEGOIV6(DataCatalog):
    """Load an image folder database to support testing image samples from MPEG-OpenimagesV6:

    .. code-block:: none
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
        seqinfo="",
        dataset_name="mpeg-oiv6-segmentation",
        ext="",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )

        if self.annotation_path:
            self._dataset = load_coco_json(
                self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
            )
        else:
            self._dataset = manual_load_data(self.imgs_folder_path, "jpg")

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

    .. code-block:: none
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
        seqinfo="seqinfo.ini",
        dataset_name="sfu-hw-object-v1",
        ext="png",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )

        self._dataset = load_coco_json(
            self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
        )

    def get_min_max_across_tensors(self):
        # from mpeg-fcvcm
        minv = -17.884761810302734
        maxv = 16.694171905517578
        return (minv, maxv)


@register_datacatalog("PANDASET")
class PANDASET(DataCatalog):
    """Load an image folder database with Detectron2 Cfg. testing image samples
    and annotations are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block:: none
        - rootdir/
            - camera
                - front_camera
                    - 00.jpg
                    - 01.jpg
                    - xx.jpg
            - annotations
                - xxxx.json TODO
    Args:
        root (string): root directory of the dataset

    """

    def __init__(
        self,
        root,
        imgs_folder="camera/front_camera",
        annotation_file=None,
        seqinfo="seqinfo.ini",
        dataset_name="pandaset",
        ext="jpg",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )

        img_lists = sorted(self.imgs_folder_path.glob(f"*.{ext}"))

        # self.data_type = "mot"
        # print(annotation_file)
        # seq_id = os.path.splitext(os.path.split(annotation_file)[1])[0]
        gt_frame_list = np.load(self.annotation_path, allow_pickle=True)[
            "gt"
        ]  # read_results(
        gt_frame_dict = {k: v for k, v in enumerate(gt_frame_list)}
        #    str(self.annotation_path), self.data_type, is_gt=True
        # )

        self._dataset = []
        self._gt_labels = gt_frame_dict
        # self._gt_ignore_labels = gt_ignore_frame_dict

        for file_name in img_lists:
            img_id = file_name.name.split(f".{ext}")[0]

            new_d = {
                "file_name": str(file_name),
                "image_id": img_id,
                "annotations": {
                    "gt": gt_frame_dict.get(int(img_id), []),
                    # "gt_ignore": gt_ignore_frame_dict.get(int(img_id), []),
                },
            }
            self._dataset.append(new_d)

    def get_min_max_across_tensors(self):
        # FIXME
        minv = -30.0
        maxv = 30.0
        return (minv, maxv)


@register_datacatalog("COCO")
class COCO(DataCatalog):
    """Load an image folder database with Detectron2 Cfg. testing image samples
    and annotations are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block:: none
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
        seqinfo="",
        dataset_name="mpeg-coco",
        ext="",
    ):
        super().__init__(
            root,
            imgs_folder=imgs_folder,
            annotation_file=annotation_file,
            seqinfo=seqinfo,
            dataset_name=dataset_name,
            ext=ext,
        )

        self._dataset = load_coco_json(
            self.annotation_path, self.imgs_folder_path, dataset_name=dataset_name
        )

    def get_min_max_across_tensors(self):
        raise NotImplementedError


@register_datacatalog("IMAGES")
class IMAGES(DataCatalog):
    """Load an image folder with images and no annotations
    (Currently this class supports none of training related operation ):

    .. code-block:: none
        - rootdir/
            - [test_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg

    Args:
        root (string): root directory of the dataset

    """

    def __init__(
        self,
        root,
        imgs_folder="test",
        annotation_file=None,
        seqinfo=None,
        dataset_name="kodak",
        ext="",
    ):
        super().__init__(
            root,
            imgs_folder,
            annotation_file,
            seqinfo,
            dataset_name,
            ext,
        )

        all_files = [
            f
            for f in sorted(self.imgs_folder_path.iterdir())
            if f.is_file() and f.suffix[1:].lower() == ext.lower()
        ]

        self._dataset = []
        for p in all_files:
            img_id = re.findall(r"[\d]+", str(Path(p).stem))
            assert len(img_id) == 1

            fw, fh = Image.open(p).size

            d = {
                "file_name": str(p),
                "height": fh,
                "width": fw,
                "image_id": img_id[0],
            }

            self._dataset.append(d)
