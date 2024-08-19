# Dataset preparation for VCM simulation of Lossless TVD and SFU-HW in CompressAI-vision

This folder includes helper scripts to prepare input YUVs for the remote inferencing pipeline to run a codec such as VCM-RS according to the VCM CTC.

This modified CompressAI-vision uses (if available) an input YUV containing only those frames to be encoded, i.e., no frames to be skipped and no trailing frames. If an input YUV not available, the fallback behaviour is to convert PNGs files into a YUV to supply to the codec.

## SFU-HW
A helper script to extract the clips for the SFU-HW usage in VCM from the original JCT-VC CTC sequences is included.

Once run, the YUVs should be under the `$VCM_TESTDATA/SFU_HW_Obj` directory, with md5sums as follows:

b4781ce438bc6f6d92dbe9a5998bf47a  BasketballDrill_832x480_50_val.yuv
6059ea62f8cf96c8ab0d8810ff866718  BasketballDrive_1920x1080_50_val.yuv
8eb2bce3548dd3ac0279be56ec1397a2  BasketballPass_416x240_50_val.yuv
71e77bf3d3cef69a9776bd21e873ba1f  BlowingBubbles_416x240_50_val.yuv
31368170ee8d935a6afb36a902366713  BQMall_832x480_60_val.yuv
a27e763cac2fbd3b8c6cd6b23fea78d0  BQSquare_416x240_60_val.yuv
097ff3cd417ecd887eb4fa9861b99e85  BQTerrace_1920x1080_60_val.yuv
9d95035e6350db9a5769b337620ebb4a  Cactus_1920x1080_50_val.yuv
fd0f841ea56fcf46b6b3c0e010549a81  Kimono_1920x1080_24_val.yuv
95a39960775205ab9e47d3722296cc34  ParkScene_1920x1080_24_val.yuv
6fe8a30fb637171a1963aafe84e36510  PartyScene_832x480_50_val.yuv
b5bb35d5a45015df61edf92fd4477ee7  RaceHorses_416x240_30_val.yuv
d147d6bd95e8ee2509c1b396f5bc44f3  RaceHorses_832x480_30_val.yuv
37cb3baf5b801edea148721903d5fb09  Traffic_2560x1600_30_val.yuv

This directory needs to also contain the SFU-HW-Objects-v3.2 ground truth (converted to JSON format), under the respective `annodations` directories in the following directory structure:

.
├── BasketballDrill_832x480_50_val
│   ├── annotations
│   └── images
├── BasketballDrive_1920x1080_50_val
│   ├── annotations
│   └── images
├── BasketballPass_416x240_50_val
│   ├── annotations
│   └── images
├── BlowingBubbles_416x240_50_val
│   ├── annotations
│   └── images
├── BQMall_832x480_60_val
│   ├── annotations
│   └── images
├── BQSquare_416x240_60_val
│   ├── annotations
│   └── images
├── BQTerrace_1920x1080_60_val
│   ├── annotations
│   └── images
├── Cactus_1920x1080_50_val
│   ├── annotations
│   └── images
├── FourPeople_1280x720_60_val
│   ├── annotations
│   └── images
├── Johnny_1280x720_60_val
│   ├── annotations
│   └── images
├── Kimono_1920x1080_24_val
│   ├── annotations
│   └── images
├── KristenAndSara_1280x720_60_val
│   ├── annotations
│   └── images
├── ParkScene_1920x1080_24_val
│   ├── annotations
│   └── images
├── PartyScene_832x480_50_val
│   ├── annotations
│   └── images
├── RaceHorses_416x240_30_val
│   ├── annotations
│   └── images
├── RaceHorses_832x480_30_val
│   ├── annotations
│   └── images
└── Traffic_2560x1600_30_val
    ├── annotations
    └── images

# TVD
Helper scripts are provided to convert from the lossless MP4 format TVD videos firstly to YUV (TVD-01 to 03) and then to produce the 500 (636 for TVD-02-1) frame clipped sequences as used in the VCM CTC.

Once complete, the YUVs need to be under the `$VCM_TESTDATA/tvd_tracking_vcm' directory, with md5sums as follows:

612b6085be31eac219a82c096633b934  TVD-01-1.yuv
c215d8f8aebf12009167ec36976a4563  TVD-01-2.yuv
a29e536ee87672d4939ddd25f6626985  TVD-01-3.yuv
aad63df298fa6401c16a36ede61e9798  TVD-02-1.yuv
221fc39367b041f983cee58623c3831f  TVD-03-1.yuv
228df384e83f6b24924c5db754d8b1ef  TVD-03-2.yuv
e5d417b9b4d6b8f8340c69f5d0ae1eb3  TVD-03-3.yuv

The following directory structure is used to contain the ground truth (`gt.txt`) files in each `gt` directory (the 'img' directories are not needed since YUV input is used):
.
├── TVD-01-1
│   ├── gt
│   └── img1
├── TVD-01-2
│   ├── gt
│   └── img1
├── TVD-01-3
│   ├── gt
│   └── img1
├── TVD-02-1
│   ├── gt
│   └── img1
├── TVD-03-1
│   ├── gt
│   └── img1
├── TVD-03-2
│   ├── gt
│   └── img1
└── TVD-03-3
    ├── gt
    └── img1

