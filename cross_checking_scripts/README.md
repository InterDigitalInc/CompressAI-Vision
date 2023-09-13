# Welcome to the crosschecking Readme

## extract files and prepare test data
1. Please first extract the provided  .tar.gz files

```
mkdir fcvcm-cfp-proposal16-package fcvcm-cfp-proposal16_feature_dumps
tar -xf fcvcm-cfp-proposal16_bitstreams_and_decoder.tar.gz --directory fcvcm-cfp-proposal16-package
tar -xf fcvcm-cfp-proposal16_feature_dumps.tar.gz --directory fcvcm-cfp-proposal16_feature_dumps
rm fcvcm-cfp-proposal16_bitstreams_and_decoder.tar.gz fcvcm-cfp-proposal16_feature_dumps.tar.gz
```

```
cd fcvcm-cfp-proposal16-package
```

The folder structure contains the source code and bitstreams to decode

2. in each dataset folder in vcm_testdata, please copy the source images from the anchor package, to enable the evaluation part of the process.

```
cp path/to/image/folder/* vcm_testdata/MPEGOIV6/images
cp path/to/image/folder/* vcm_testdata/SFUHW/images
cp path/to/image/folder/* vcm_testdata/TVD/images
cp path/to/image/folder/* vcm_testdata/HiEve/images
```

## Install
3. create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```


4. From a cuda capable machine, load cuda environment (necessary for task models)

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



8. From a machine equipped with microsoft excel:
- open the generate csv files
- open the provided result file BLABLA.xls at the root of this package

copy and paste relevant sections to the template


9. Check feature dumps
...



10. Optional, crosscheck encoder runtimes
...