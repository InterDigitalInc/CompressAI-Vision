# Scripts to collect results and generate CVS sheets readdy for MPEG common test conditions

To generate the csv related to a dataset for the OIv6 dataset for instance, modify the paths in the following command and run:



### Example commands to collect the metrics of split-inference pipelines 

```
python /path-to-compressai_vision/scripts/metrics/gen_mpeg_cttc_csv.py  --dataset_name OIV6 \
                                                                        --dataset_path /path/to/fcm_testdata/mpeg-oiv6 \
                                                                        --result_path output_dir/split-inference-image/{codec_name}_{exp_name}/MPEGOIV6
```

For a video dataset, e.g., SFU:

```
python /path-to-compressai_vision/scripts/metrics/gen_mpeg_cttc_csv.py  --dataset_name SFU \ 
                                                                        --dataset_path /path/to/fcm_testdata/SFU_HW_Obj \
                                                                        --result_path output_dir/split-inference-video/{codec_name}_{exp_name}/SFUHW
```

### Example commands to collect the metrics of remote-inference pipelines 

```
python /path-to-compressai_vision/scripts/metrics/gen_mpeg_cttc_csv.py  --dataset_name SFU \
                                                                        --dataset_path /path/to/vcm_testdata/tvd_tracking \
                                                                        --result_path output_dir/remote-inference-video/{codec_name}_{exp_name}/MPEGTVDTRACKING/ \
                                                                        --nb_operation_points 6 \
                                                                        --remote_inference \
                                                                        --curve_fit
```
