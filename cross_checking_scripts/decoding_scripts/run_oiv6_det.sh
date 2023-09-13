#! /usr/bin/env bash

RUN="sequential" # "gnu_parallel" or "sequential" or "slurm"
INPUT_DIR="/data/datasets/MPEG-FCVCM/vcm_testdata" # needed for NN_PART2
BITSTREAM_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs/cfp_test" 

#################################################################
# CODEC_PARAMS="++pipeline.codec.encode_only=True" -> run only encode
# CODEC_PARAMS="++pipeline.codec.decode_only=True" -> run only decode
# CODEC_PARAMS="" -> run encode + decode
CODEC_PARAMS="++pipeline.codec.decode_only=True" 
EXPERIMENT="_gen_bitstreams_v1"
QPS=`echo "8 12"` 
DEVICE="cuda"
#################################################################

# total number of jobs = 6
if [[ ${RUN} == "gnu_parallel" ]]; then
    MAX_PARALLEL=6 
    run_scripts () {
        sem -j $MAX_PARALLEL bash $1
    }
    export -f run_scripts
elif [[ ${RUN} == "slurm" ]]; then
    run_scripts () {
        sbatch --gpus 1 --reservation=deepvideo --job-name=oiv6_det_decode $1
    }
    export -f run_scripts
else
    run_scripts () {
        bash $1
    }
    export -f run_scripts
fi


# DECODE + NN_PART2
for QP in ${QPS}
do
    echo RUN: ${RUN}, Input Dir: ${INPUT_DIR}, Bitstream Dir: ${BITSTREAM_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}, CODEC_PARAMS: ${CODEC_PARAMS}
    run_scripts "../mpeg_cfp_oiv6_detection.sh ${INPUT_DIR} ${BITSTREAM_DIR} ${EXPERIMENT} ${DEVICE} ${QP} ${CODEC_PARAMS}"
done

# GENERATE CSV
if [[ ${RUN} == "gnu_parallel" ]]; then
    sem --wait
    bash gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-detection
elif [[ ${RUN} == "slurm" ]]; then
    sbatch --dependency=singleton --job-name=oiv6_det_decode  gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-detection
else
    bash gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-detection
fi
