# Welcome to the crosschecking Readme

## extract files and prepare test data
1. Please first extract the provided  .tar.gz files

```
mkdir fcvcm-cfp-proposal16_package fcvcm-cfp-proposal16_feature_dumps
tar -xf fcvcm-cfp-proposal16_bitstreams_and_decoder.tar.gz --directory fcvcm-cfp-proposal16_package
tar -xf fcvcm-cfp-proposal16_feature_dumps.tar.gz --directory fcvcm-cfp-proposal16_feature_dumps
rm fcvcm-cfp-proposal16_bitstreams_and_decoder.tar.gz fcvcm-cfp-proposal16_feature_dumps.tar.gz
```

Note, as our first archive has misplaced bitstreams, we provide updated
- fcvcm-cfp-proposal16_bitstreams_and_decoder-late-09-14.tar.gz
- fcvcm-cfp-proposal16_feature_dumps-late-09-14.tar.gz
Please change above paths accordingly if using the updated package. 

```
cd fcvcm-cfp-proposal16-package
```

You should see:
- compressai-fcvmc: source code and scripts
- split-inference-image: bitstreams for openimagev6
- split-inference-video: bitstreams for sfuhw, tvd and hieve
- vcm_testdata 


2. the vcm_testdata folder contains annotations and sequence parameters like in MPEG FCVCM anchor package. As the images are too voluminous, please copy the source images from the anchor package to vcm_testdata to enable the evaluation part of the process.

For openimage:
```
cp your/path/to/image/MPEGOIV6/images/*.jpg vcm_testdata/MPEGOIV6/images
```

each sequence of the video datasets (we provide an example for each)
```
cp your/path/to/image/folder/13/img1/*.png vcm_testdata/HiEve_pngs/13/img1/
```
```
cp your/path/to/image/folder/13/img1/*.png vcm_testdata/HiEve_pngs/13/img1/
```
```
cp your/path/to/image/folder/BasketballDrill_832x480_50_val/images/*.png vcm_testdata/SFU_HW_Obj/BasketballDrill_832x480_50_val/images/img1/
```


## Install
3. create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

4. From a cuda capable machine, load cuda environment (necessary for task models only)

```
bash compressai-fcvcm/scripts/env_cuda.sh 11.8
```

6. install this pacakes, the tassk models and related dependencies
```
bash compressai-fcvcm/scripts/install.sh
```

# Run cross-check evaluation
7. Run crosscheck scripts: decode and evaluat.
Default settings: sequential runs on CPU:
```
bash compressai-fcvcm/cross_checking_scripts/decoding_scripts/run_oiv6_det.sh
bash compressai-fcvcm/cross_checking_scripts/decoding_scripts/run_oiv6_seg.sh
bash compressai-fcvcm/cross_checking_scripts/decoding_scripts/run_sfu.sh
bash compressai-fcvcm/cross_checking_scripts/decoding_scripts/run_tvd.sh
bash compressai-fcvcm/cross_checking_scripts/decoding_scripts/run_hieve.sh
```

Note: 
- you may have to check paths at the top of the scripts if you use testdata stored elsewhere
- You can parrallelize with gnu or slurm system, check script headers for options

This will populate the existing folders split-inference-video and split-inference-image with the evaluation results

A .csv is generated at each location 
- split-inference-image/cfp_codec/MPEGOIV6/mpeg-oiv6-detection/final_OIV6.csv
- split-inference-image/cfp_codec/MPEGOIV6/mpeg-oiv6-segmentation/final_OIV6.csv
- split-inference-video/cfp_codec/MPEGHIEVE/final_HIEVE.csv
- split-inference-video/cfp_codec/MPEGTVDTRACKING/final_TVD.csv
- split-inference-video/cfp_codec/SFUHW/final_SFU.csv

To generate the Per-Class results, please run from the root:
```
python compressai-fcvcm/utils/compute_overall_mot.py  -a --result_path=split-inference-video/cfp_codec/MPEGHIEVE --dataset_path=vcm_testdata/HiEve_pngs --class_to_compute=HIEVE-720P
```
```
python compressai-fcvcm/utils/compute_overall_mot.py  -a --result_path=split-inference-video/cfp_codec/MPEGHIEVE --dataset_path=vcm_testdata/HiEve_pngs --class_to_compute=HIEVE-1080P
```
```
python compressai-fcvcm/utils/compute_overall_mot.py  -a --result_path=split-inference-video/cfp_codec/MPEGTVDTRACKING --dataset_path=vcm_testdata/tvd_tracking --class_to_compute=TVD
```
```
python compressai-fcvcm/utils/compute_overall_map.py  -a --result_path=split-inference-video/cfp_codec/SFUHW --dataset_path=vcm_testdata/SFU_HW_Obj --class_to_compute=CLASS-AB
```
```
python compressai-fcvcm/utils/compute_overall_map.py  -a --result_path=split-inference-video/cfp_codec/SFUHW --dataset_path=vcm_testdata/SFU_HW_Obj --class_to_compute=CLASS-C
```
```
python compressai-fcvcm/utils/compute_overall_map.py  -a --result_path=split-inference-video/cfp_codec/SFUHW --dataset_path=vcm_testdata/SFU_HW_Obj --class_to_compute=CLASS-D
```

That should create the following files
- split-inference-video/cfp_codec/MPEGHIEVE/HIEVE-720P.csv
- split-inference-video/cfp_codec/MPEGHIEVE/HIEVE-1080P.csv
- split-inference-video/cfp_codec/MPEGTVDTRACKING/TVD.csv
- split-inference-video/cfp_codec/SFUHW/CLASS-AB.csv
- split-inference-video/cfp_codec/SFUHW/CLASS-C.csv
- split-inference-video/cfp_codec/SFUHW/CLASS-D.csv



8. From a machine equipped with microsoft excel:
- open the generated csv files at the root of each dataset result 
- open the provided result file fcvcm-cfp-proposal16.xlsm at the root of this package

copy and paste relevant sections to the template.

Note: for Per-Class results, copy only the accuracy metrics column, the bitrates are directly calculated in excel 


9. Check feature dumps located in the unzipped fcvcm-cfp-proposal16_feature_dumps

