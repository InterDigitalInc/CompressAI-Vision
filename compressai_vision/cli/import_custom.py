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

"""Use this stub for adding new cli commands
"""
import os

from pathlib import Path

from .tools import makeVideoThumbnails

possible_types = [
    "oiv6-mpeg-v1",  # as provided by nokia
    "tvd-object-tracking-v1",  # TVD
    "tvd-image-v1",  # TVD
    "sfu-hw-objects-v1",  # SFU-HW
    "flir-mpeg-v1",
    "flir-image-rgb-v1",  # FLIR
]


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "import-custom",
        parents=parents,
        help="import some popular custom datasets into fiftyone",
    )
    req_group = subparser.add_argument_group("required arguments")
    req_group.add_argument(
        "--dataset-type",
        action="store",
        type=str,
        required=True,
        default=None,
        help="dataset type, possible values: " + ",".join(possible_types),
    )
    req_group.add_argument(
        "--dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="root directory of the dataset",
    )
    mpeg_group = subparser.add_argument_group("arguments for oiv6-mpeg")
    mpeg_group.add_argument(
        "--datadir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory where all datasets are downloaded by default (optional)",
    )
    mpeg_group.add_argument(
        "--mock", action="store_true", default=False, help="debugging switch: don't use"
    )
    mpeg_group.add_argument(
        "--use-vcm",
        action="store_true",
        default=True,
        help="Use mpeg-vcm files bundled with compressai-vision",
    )


def main(p):  # noqa: C901
    assert (
        p.dataset_type in possible_types
    ), "dataset-type needs to be one of these:" + str(possible_types)
    print("importing fiftyone")
    import fiftyone as fo
    from compressai_vision import patch  # required by tvd-image-v1 import

    # from fiftyone import ViewField as F

    print("fiftyone imported")
    try:
        dataset = fo.load_dataset(p.dataset_type)
        assert dataset is not None  # dummy
    except ValueError:
        pass
    else:
        print(
            "WARNING: dataset %s already exists: will delete and rewrite"
            % (p.dataset_type)
        )

    # oiv-mpeg-v1 doesn't need to --dir (p.dir) since it downloads the file itself
    if p.dataset_type == "oiv6-mpeg-v1":
        # this is the most "ancient" part in this library
        # (all started by trying to import oiv-mpeg-v1)
        from .auto import main

        # see func add_subparser up there
        # main is using the parameters in mpeg_group
        main(p)
        return

    # rest of the importers require the user to download the files themselves
    assert p.dir is not None, "please provide root directory with the --dir argument"
    p.dir = os.path.expanduser(p.dir)  # correct path in the case user uses POSIX "~"
    assert os.path.isdir(p.dir), "can find directory " + p.dir

    print()
    print("Importing custom dataset into fiftyone")
    print()
    print("Dataset type           : ", p.dataset_type)
    print("Dataset root directory : ", p.dir)
    print()

    if p.dataset_type == "sfu-hw-objects-v1":
        if not p.y:
            input("press enter to continue.. ")
            print()
        from compressai_vision.conversion.sfu_hw_objects_v1 import (
            register,
            video_convert,
        )

        video_convert(p.dir)
        register(p.dir)  # dataset persistent
        """NOTE: not required for this dataset
        print()
        print("Will create thumbnails for fiftyone app visualization")
        print("for your convenience, Sir")
        if not p.y:
            input("press enter to continue.. (or CTRL-C to abort)")
            print()
        dataset=fo.load_dataset("sfu-hw-objects-v1")
        makeVideoThumbnails(dataset, force=True)
        """

    elif p.dataset_type == "tvd-object-tracking-v1":
        if not p.y:
            input("press enter to continue.. ")
            print()
        from compressai_vision.conversion.tvd_object_tracking_v1 import register

        res=register(p.dir)  # dataset persistent
        if res is not None:
            return 2
        print()
        print("Will create thumbnails for fiftyone app visualization")
        print("for your convenience, Sir")
        if not p.y:
            input("press enter to continue.. (or CTRL-C to abort)")
            print()
        dataset = fo.load_dataset("tvd-object-tracking-v1")
        makeVideoThumbnails(dataset, force=True)

    elif p.dataset_type == "tvd-image-v1":
        print(
            """
        After extracting tencent zipfiles:

        TVD_Instance_Segmentation_Annotations.zip
        TVD_Object_Detection_Dataset_and_Annotations.zip

        You should have this directory structure:
        /path/to/
            TVD_Object_Detection_Dataset_And_Annotations/
                tvd_detection_validation_bbox.csv
                tvd_detection_validation_labels.csv
                tvd_label_hierarchy.json
                tvd_object_detection_dataset/ (IMAGES)
            tvd_segmentation_validation_bbox.csv
            tvd_segmentation_validation_labels.csv
            tvd_segmentation_validation_masks.csv
            tvd_validation_masks/ (SEGMASKS)
        """
        )
        print("you have defined /path/to = ", p.dir)
        print()
        print(
            """
        OpenImageV6 formatted files and directory structures will be in

        /path/to/TVD_images_detection_v1
        /path/to/TVD_images_segmentation_v1
        """
        )
        if not p.y:
            input("press enter to continue.. ")
            print()

        mainpath = Path(p.dir)
        # bbox
        bbox_path = mainpath / "TVD_Object_Detection_Dataset_And_Annotations"
        bbox_validation_csv_file = bbox_path / "tvd_detection_validation_labels.csv"
        bbox_csv_file = bbox_path / "tvd_detection_validation_bbox.csv"
        img_dir = bbox_path / "tvd_object_detection_dataset"
        bbox_target_dir = mainpath / "TVD_images_detection_v1"
        # seg
        seg_validation_csv_file = mainpath / "tvd_segmentation_validation_labels.csv"
        seg_csv_file = mainpath / "tvd_segmentation_validation_bbox.csv"
        seg_mask_csv_file = mainpath / "tvd_segmentation_validation_masks.csv"
        seg_data_dir = mainpath / "tvd_validation_masks"
        seg_target_dir = mainpath / "TVD_images_segmentation_v1"
        #
        from compressai_vision.conversion.mpeg_vcm import MPEGVCMToOpenImageV6

        # detections
        MPEGVCMToOpenImageV6(
            validation_csv_file=str(bbox_validation_csv_file),
            bbox_csv_file=str(bbox_csv_file),
            output_directory=str(bbox_target_dir),
            data_dir=str(img_dir),
            link=True,
            # link=False,
            verbose=True,
        )

        # segmentations
        MPEGVCMToOpenImageV6(
            validation_csv_file=str(seg_validation_csv_file),
            bbox_csv_file=str(seg_csv_file),
            segmentation_csv_file=str(seg_mask_csv_file),
            output_directory=str(seg_target_dir),
            mask_dir=str(seg_data_dir),
            data_dir=str(img_dir),
            # mask_dir: str = None,
            link=True,
            # link=False,
            verbose=True,
            append_mask_dir="0"
            # since the dir structure provided is erroneous
            # create the labels/masks directory ourselves
            # and link from labels/masks/0 --> provided segmask dir
        )

        name = "tvd-image-detection-v1"
        print("\nRegistering", name)
        try:
            fo.delete_dataset(name)
        except ValueError:
            pass
        else:
            print("WARNING: deleted pre-existing", name)
        dataset = fo.Dataset.from_dir(
            name=name,
            dataset_dir=str(bbox_target_dir),
            dataset_type=fo.types.dataset_types.OpenImagesV6Dataset,
            # label_types=("detections", "classifications", "relationships", "segmentations"),
            label_types=("detections", "classifications"),
            load_hierarchy=False,
        )
        dataset.persistent = True

        name = "tvd-image-segmentation-v1"
        print("\nRegistering", name)
        try:
            fo.delete_dataset(name)
        except ValueError:
            pass
        else:
            print("WARNING: deleted pre-existing", name)
        dataset = fo.Dataset.from_dir(
            name=name,
            dataset_dir=str(seg_target_dir),
            dataset_type=fo.types.dataset_types.OpenImagesV6Dataset,
            # label_types=("detections", "classifications", "relationships", "segmentations"),
            label_types=("detections", "segmentations", "classifications"),
            load_hierarchy=False,
        )
        dataset.persistent = True

    elif p.dataset_type == "flir-mpeg-v1":
        name = "flir-mpeg-detection-v1"
        print(
            """
        After extraing mpeg-vcm provided zipfile, you should have this directory/file structure:

        /path/to/
        |   ├── anchor_results
        │   ├── FLIR_anchor_vtm12_bitdepth10.xlsx
        │   └── VCM-reporting-template-FLIR_vtm12_d10.xlsm
        ├── dataset
        │   ├── coco_format_json_annotation
        │   │   ├── FLIR_val_thermal_coco_format_jpg.json
        │   │   ├── FLIR_val_thermal_coco_format_png.json
        │   │   └── Two files differ only in image file format whithin the file, and the rest are the same..txt
        │   ├── fine_tuned_model
        │   │   └── model_final.pth
        │   └── thermal_images [300 entries exceeds filelimit, not opening dir]
        ├── mAP_coco.py
        └── Readme.txt

        You provided /path/to = %s
            """
            % (p.dir)
        )
        if not p.y:
            input("press enter to continue.. ")
            print()
        print("\nRegistering", name)
        try:
            fo.delete_dataset(name)
        except ValueError:
            pass
        else:
            print("WARNING: deleted pre-existing", name)
        # https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#basic-recipe
        dataset = fo.Dataset.from_dir(
            name=name,
            dataset_type=fo.types.COCODetectionDataset,
            data_path=os.path.join(p.dir, "dataset", "thermal_images"),
            labels_path=os.path.join(
                p.dir,
                "dataset",
                "coco_format_json_annotation",
                "FLIR_val_thermal_coco_format_jpg.json",
            ),
        )
        dataset.persistent = True

    elif p.dataset_type == "flir-image-rgb-v1":
        name = p.dataset_type
        print(
            """
        After extracting

        FLIR_ADAS_v2.zip

        You should have the following (COCO-formatted) directory/file structure:

        images_rgb_train/
            coco.json
            data/ [IMAGES]
        images_rgb_val/
            coco.json
            data/ [IMAGES]
        images_thermal_train/
            coco.json
            data/ [IMAGES]
        images_thermal_val/
            ...
        video_rgb_test/
            ...
        video_thermal_test/
            ...
        rgb_to_thermal_vid_map.json

        Will import
            %s/images_rgb_train
            into dataset flir-image-rgb-v1
        """
            % (p.dir)
        )
        if not p.y:
            input("press enter to continue.. ")
            print()
        print("\nRegistering", name)
        try:
            fo.delete_dataset(name)
        except ValueError:
            pass
        else:
            print("WARNING: deleted pre-existing", name)
        dataset_dir = os.path.join(p.dir, "images_rgb_train")
        dataset = fo.Dataset.from_dir(
            name=name,
            dataset_type=fo.types.COCODetectionDataset,
            data_path=os.path.join(dataset_dir),  # , "data"),
            labels_path=os.path.join(dataset_dir, "coco.json"),
            # image_ids = [] # TODO
        )
        dataset.persistent = True

    print("HAVE A NICE DAY!")
