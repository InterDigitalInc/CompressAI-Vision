#!/usr/bin/env python
"""
Extract SFU crops in YUV format from the original sequences
"""

import os
from sfu_dicts import seq_dict, res_dict, fr_dict

# Edit following paths as needed:
SRC_YUV_DIR = os.path.expandvars("$SEQUENCE_DIR/x1100")
DST_YUV_DIR = os.path.expandvars("$VCM_TESTDATA/SFU_HW_Obj-v3.2_supplied_jsons")


BYTES_PER_SAMPLE = 1.5


def generate_yuv_clip(
    sequence, sequence_name, width, height, frames_to_be_encoded, frame_skip
):
    src_yuvs = os.listdir(SRC_YUV_DIR)
    src_yuvs = [
        sy
        for sy in src_yuvs
        if sy.startswith(sequence_name.replace("_val", "").replace("Kimono", "Kimono1"))
    ]
    assert len(src_yuvs) == 1
    src_yuv = os.path.join(SRC_YUV_DIR, src_yuvs[0])
    dst_yuv = os.path.join(DST_YUV_DIR, f"{sequence_name}.yuv")
    print(src_yuv, dst_yuv)

    frame_size = int(width * height * 1.5)
    offset = frame_size * frame_skip
    length = frame_size * frames_to_be_encoded

    with open(src_yuv, "rb") as f:
        f.seek(offset)
        bytes = f.read(length)

    with open(dst_yuv, "wb") as f:
        f.write(bytes)


def main():
    for sequence, (clas, sequence_name) in seq_dict.items():
        width, height = res_dict[clas]
        intra_period, frame_rate, frames_to_be_encoded, frame_skip = fr_dict[sequence]
        generate_yuv_clip(
            sequence, sequence_name, width, height, frames_to_be_encoded, frame_skip
        )


main()
