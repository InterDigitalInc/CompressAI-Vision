#!/usr/bin/env python
"""
Extract SFU crops in YUV format from the original sequences
"""

import os

SEQ_DICT = {
    "TVD-01-1": ["TVD-01", 3000, 50, 8, 1500, 500],
    "TVD-01-2": ["TVD-01", 3000, 50, 8, 2000, 500],
    "TVD-01-3": ["TVD-01", 3000, 50, 8, 2500, 500],
    "TVD-02-1": ["TVD-02", 636, 50, 10, 0, 636],
    "TVD-03-1": ["TVD-03", 2334, 50, 10, 0, 500],
    "TVD-03-2": ["TVD-03", 2334, 50, 10, 500, 500],
    "TVD-03-3": ["TVD-03", 2334, 50, 10, 1000, 500],
}

SRC_YUV_DIR = os.path.expandvars("$VCM_TESTDATA/tvd_tracking_lossless")
DST_YUV_DIR = os.path.expandvars("$VCM_TESTDATA/tvd_tracking_vcm")

ALL_SRC_FILES = os.listdir(SRC_YUV_DIR)
ALL_SRC_YUVS = [sf for sf in ALL_SRC_FILES if sf.endswith(".yuv")]
# BYTES_PER_SAMPLE = 1.5

os.makedirs(DST_YUV_DIR, exist_ok=True)


def generate_yuv_clip(
    sequence, src_seq, width, height, frames_to_be_encoded, frame_skip, bit_depth
):
    src_yuvs = [sy for sy in ALL_SRC_YUVS if sy.startswith(src_seq)]

    assert len(src_yuvs) == 1
    src_yuv = os.path.join(SRC_YUV_DIR, src_yuvs[0])
    dst_yuv = os.path.join(DST_YUV_DIR, f"{sequence}.yuv")
    imgs_subfolder = os.path.join(DST_YUV_DIR, sequence, "imgs")
    os.makedirs(
        imgs_subfolder, exist_ok=True
    )  # Needed for CompressAI-Vision folder presence check
    print(src_yuv, dst_yuv)

    bytes_per_sample = (bit_depth + 7) >> 3
    frame_size = int(width * height * 1.5 * bytes_per_sample)
    offset = frame_size * frame_skip
    length = frame_size * frames_to_be_encoded

    with open(src_yuv, "rb") as f:
        f.seek(offset)
        bytes = f.read(length)

    with open(dst_yuv, "wb") as f:
        f.write(bytes)


def main():
    for sequence, params in SEQ_DICT.items():
        width, height = 1920, 1080
        # intra_period, frame_rate = 64, 50
        (
            src_seq,
            total_frames,
            frame_rate,
            bit_depth,
            frame_skip,
            frames_to_be_encoded,
        ) = params

        generate_yuv_clip(
            sequence,
            src_seq,
            width,
            height,
            frames_to_be_encoded,
            frame_skip,
            bit_depth,
        )


if __name__ == "__main__":
    main()
