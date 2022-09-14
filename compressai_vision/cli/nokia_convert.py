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

"""cli nokia_convert functionality
"""
import os


def main(p):
    # compressai_vision
    from compressai_vision.conversion import nokiaBSToOpenImageV6  # imageIdFileList
    from compressai_vision.tools import pathExists

    assert p.target_dir is not None, "please give target_dir"
    assert (
        p.dir is not None
    ), "please specify OpenImageV6 source directory for images (and masks)"
    if pathExists(p.target_dir):
        print(
            "FATAL: target directory %s exists already. Please remove it first"
            % (p.target_dir)
        )
        return
    assert pathExists(p.dir), "directory " + p.dir + " does not exist"
    assert (
        p.label is not None
    ), "please provide image level labels --label ('detection_validation_labels_5k.csv')"
    image_dir = os.path.join(p.dir, "data")
    mask_dir = os.path.join(p.dir, "labels", "masks")
    assert pathExists(image_dir), "directory " + image_dir + " does not exist"
    if p.bbox is not None:
        assert pathExists(p.bbox), "file " + p.bbox + " does not exist"
        p.bbox = os.path.expanduser(p.bbox)
    if p.mask is not None:
        assert pathExists(p.mask), "file " + p.mask + " does not exist"
        assert pathExists(mask_dir), "directory " + mask_dir + " does not exist"
        p.mask = os.path.expanduser(p.mask)

    print()
    assert p.lists is not None, "a list file (.lst) required --lists"
    fnames = p.lists.split(",")
    assert len(fnames) == 1, "please specify exactly one list file for nokia convert"
    fname = fnames[0]
    print("Using list file         :    ", fname)
    print("Images (and masks) from :    ", p.dir)
    print("  --> from: ", image_dir)
    print("  --> from:", mask_dir)
    print("Image-level labels from :    ", p.label)
    print("Detections (bboxes)from :    ", p.bbox)
    print("Segmasks from           :    ", p.mask)
    print("Final OIV6 format in    :    ", p.target_dir)
    if not p.y:
        input("press enter to continue.. ")
    nokiaBSToOpenImageV6(
        validation_csv_file=os.path.expanduser(p.label),
        list_file=os.path.expanduser(fname),
        bbox_csv_file=p.bbox,
        segmentation_csv_file=p.mask,
        output_directory=os.path.expanduser(p.target_dir),
        data_dir=os.path.expanduser(image_dir),
        mask_dir=os.path.expanduser(mask_dir),
    )
    print("nokia convert ready, please check", p.target_dir)
