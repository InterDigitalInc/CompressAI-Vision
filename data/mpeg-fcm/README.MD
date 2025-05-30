# FCM test data

The test configurations for 5 use-cases split in 4 test datasets are provided.
The following
Original test images and ground truths can be copied into the relevant folders following the structure below:

```bash
fcm_testdata
├── HiEve_pngs
│   ├── 13
│   │   ├── gt
│   │   ├── img1
│   │   └── seqinfo.ini
│   ├── 16
│   │   ├── gt
│   │   ├── img1
│   │   └── seqinfo.ini
│   ├── ...
├── SFU_HW_Obj
│   ├── BQMall_832x480_60_val
│   │   ├── annotations
│   │   ├── images
│   │   └── seqinfo.ini
│   ├── BQSquare_416x240_60_val
│   │   ├── annotations
│   │   ├── images
│   │   └── seqinfo.ini
│   ├── ...
├── mpeg-oiv6
│   ├── annotations
│   │   ├── mpeg-oiv6-detection-coco.json
│   │   └── mpeg-oiv6-segmentation-coco.json
│   └── images
└── tvd_tracking
    ├── MD5.txt
    ├── TVD-01
    │   ├── gt
    │   ├── img1
    │   └── seqinfo.ini
    ├── TVD-02
    │   ├── gt
    │   ├── img1
    │   └── seqinfo.ini
    └── ...
```
Please copy the source image files into each "images" or "img1" folder.
The ground-truth json files are provided for mpeg-oiv6 since we use concatenated data into one file per use-case.
Please download them from https://dspub.blob.core.windows.net/mpeg-fcm/dataset/mpeg-oiv6-dataset.tar.gz

Finally use the path to fcm_testdata as dataset.config.root in your configuration


# FPN sizes
FPN sizes correspond to the information necessary for reconstrucing tensors in their original shape at the decoder,
when the information is not provided in the bitstreams (like for Feature Anchors).
```bash
.
├── MPEGHIEVE
│   └── fpn-sizes
├── MPEGOIV6
│   └── fpn-sizes
│       ├── mpeg-oiv6-detection
│       └── mpeg-oiv6-segmentation
├── MPEGTVDTRACKING
│   └── fpn-sizes
├── SFUHW
│   └── fpn-sizes
```