#! /usr/bin/env bash

RUN="slurm" # "gnu_parallel" or "sequential" or "slurm"
INPUT_DIR="/data/datasets/MPEG-FCVCM/vcm_testdata" # needed for NN_PART2
BITSTREAM_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs/cfp_test" 

#################################################################
# CODEC_PARAMS="++pipeline.codec.encode_only=True" -> Only Encode
# CODEC_PARAMS="++pipeline.codec.decode_only=True" -> Only Decode
# CODEC_PARAMS="" -> Encode + Decode
CODEC_PARAMS="++pipeline.codec.decode_only=True" 
EXPERIMENT="_gen_bitstreams_v1"
QPS=`echo "8 12"`
DEVICE="cuda"
#################################################################

# total number of jobs = 30
if [[ ${RUN} == "gnu_parallel" ]]; then
    MAX_PARALLEL=30 
    run_scripts () {
        sem -j $MAX_PARALLEL bash $1
    }
    export -f run_scripts
elif [[ ${RUN} == "slurm" ]]; then
    run_scripts () {
        sbatch --gpus 1 --reservation=deepvideo --job-name=hieve_decode $1
    }
    export -f run_scripts
else
    run_scripts () {
        bash $1
    }
    export -f run_scripts
fi

for SEQ in \
            '13' \
            '16' \
            '2' \
            '17' \
            '18'
do
    for QP in ${QPS}
    do
        echo RUN: ${RUN}, Input Dir: ${INPUT_DIR}, Bitstream Dir: ${BITSTREAM_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}, SEQ: ${SEQ}, CODEC_PARAMS: ${CODEC_PARAMS}
        run_scripts "../mpeg_cfp_hieve.sh ${INPUT_DIR} ${BITSTREAM_DIR} ${EXPERIMENT} ${DEVICE} ${QP} ${SEQ} ${CODEC_PARAMS}"
    done
done

# GENERATE CSV
if [[ ${RUN} == "gnu_parallel" ]]; then
    sem --wait
    bash gen_csv.sh HIEVE ${BITSTREAM_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGHIEVE/
elif [[ ${RUN} == "slurm" ]]; then
    sbatch --dependency=singleton --job-name=hieve_decode  gen_csv.sh HIEVE ${BITSTREAM_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGHIEVE/
else
    bash gen_csv.sh HIEVE ${BITSTREAM_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGHIEVE/
fi


