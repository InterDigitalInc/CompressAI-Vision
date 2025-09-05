# Evaluation scripts

This folder contains exemplary scripts to run the split and remote inference pipelines on the supported datasets and corresponding datasets. It is organized by use case, each bash scripts can be run either in split inference or remote inference mode. 

## Run the different bash scripts including compression
Here is a template command to run a split of the HiEve dataset, directly using HEVC refenrence software HM:
compressai-remote-inference
```
bash ${script_path} \
    --testdata "${fcm_testdata}" \
    --pipeline "${pipeline}"
    --inner_codec "${inner_codec_path}" \
    --output_dir "${output_dir}" \
    --exp_name "${exp_name}" 
    --device "${device}" 
    --qp "${qp}" 
    --seq_name "${seq_name}" \
    --extra_params "${pipeline_params}" \
    --config "${config_name}"
```
with 
- `${script_path}`: one of the bash scripts within that directory, e.g., mpeg_oiv6_vtm.sh
- `${pipeline}`: split/remote, default="split"
- `${fcm_testdata}`: path to the test dataset, e.g. path/to/fcm_testdata/
- `${inner_codec_path}`: path to core codec, e.g. /path/to/VTM_repo (that contains subfolders bin/ cfg/ ...) 
- `${output_dir}`: root of output folder for logs and artifacts
- `${exp_name}`: name of the experiments to locate/label outputs
- `${device}`: cuda or cpu
- `${qp}`: quality level, depends on the inner codec
- `${seq_name}`: sequence name as used in testdata root folder. E.g., "Traffic_2560x1600_30_val" in sfu_hw_obj
- `${pipeline_params}`: additional parameters to override default configs (pipeline/codec/evaluation...). E.g., 
```
++codec.encoder_config.stash_outputs=True \
++codec.encoder_config.chroma_format='420' \
++codec.encoder_config.input_bitdepth=8 \
++codec.encoder_config.output_bitdepth=10"
```

## Test your environment
In addition, the following script is provided to test your environment
```
bash default_vision_performances.sh --command [entry_cmd] --testdata [/path/to/dataset] --device [device]"
```
it runs the evaluation of the performance of the vision models without compression of input video or intermediate data. In both pipeline types, the input content is passed to the decoder without compression. 


