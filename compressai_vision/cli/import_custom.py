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

possible_types = ["sfu-hw-objects-v1", "tvd-object-tracking-v1", "tvd-image-v1"]


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
        required=True,
        default=None,
        help="root directory of the dataset",
    )


def main(p):
    p.dir = os.path.expanduser(p.dir)  # correct path in the case user uses POSIX "~"
    assert (
        p.dataset_type in possible_types
    ), "dataset-type needs to be one of these:" + str(possible_types)
    assert os.path.isdir(p.dir), "can find directory " + p.dir

    print("importing fiftyone")
    import fiftyone as fo

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

    print()
    print("Importing a custom video format into fiftyone")
    print()
    print("Dataset type           : ", p.dataset_type)
    print("Dataset root directory : ", p.dir)
    print()
    
    # implement different (custom) datasets here
    if p.dataset_type == "sfu-hw-objects-v1":
        if not p.y:
            input("press enter to continue.. ")
            print()
        from compressai_vision.conversion.sfu_hw_objects_v1 import (
            register,
            video_convert,
        )
        video_convert(p.dir)
        register(p.dir)

    elif p.dataset_type == "tvd-object-tracking-v1":
        if not p.y:
            input("press enter to continue.. ")
            print()
        from compressai_vision.conversion.tvd_object_tracking_v1 import register
        register(p.dir)

    elif p.dataset_type == "tvd-image-v1":
        print("""
        When extracting tencent zipfiles

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
        """)
        print("you have defined /path/to = ", p.dir)
        print()
        print("""
        OpenImageV6 formatted dir structure will be in

        /path/to/TVD_images_detection_v1
        /path/to/TVD_images_segmentation_v1
        """)
        if not p.y:
            input("press enter to continue.. ")
            print()

        mainpath=Path(p.dir)
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
            output_directory = str(bbox_target_dir),
            data_dir = str(img_dir),
            link=True,
            # link=False,
            verbose=True
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
            verbose=True
        )

        name="tvd-image-detection-v1"
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
            load_hierarchy=False
        )

        name="tvd-image-segmentation-v1"
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
            load_hierarchy=False
        )
        print("HAVE A NICE DAY!")
