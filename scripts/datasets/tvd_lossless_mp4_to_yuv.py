#!/usr/bin/env python
import os
from pathlib import Path

TVD_MP4_DIR = Path(os.path.expandvars("${VCM_TESTDATA}")) / "tvd_tracking_lossless"

TVD_YUV_DIR = Path(os.path.expandvars("${VCM_TESTDATA}")) / "tvd_tracking_lossless"


ALL_FILES = os.listdir(TVD_MP4_DIR)
MP4_FILES = [af for af in ALL_FILES if af.endswith(".mp4")]

print("Found:", MP4_FILES)

for seq in ["TVD-01", "TVD-02", "TVD-03"]:
    src_files = [mp4f for mp4f in MP4_FILES if mp4f.startswith(seq)]
    src_file = src_files[0]
    cmd = f"ffmpeg -i {TVD_MP4_DIR}/{src_file} {TVD_YUV_DIR}/{seq}.yuv"
    print(cmd)
    os.system(cmd)
print("DONE")
