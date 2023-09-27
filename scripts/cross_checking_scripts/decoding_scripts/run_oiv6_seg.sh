#! /usr/bin/env bash

RUN="sequential" # "gnu_parallel" or "sequential" or "slurm"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INPUT_DIR="${SCRIPT_DIR}/../../../vcm_testdata" # needed for NN_PART2
BITSTREAM_DIR="${SCRIPT_DIR}/../../../"

#################################################################
# CODEC_PARAMS="++pipeline.codec.encode_only=True" -> run only encode
# CODEC_PARAMS="++pipeline.codec.decode_only=True" -> run only decode
# CODEC_PARAMS="" -> run encode + decode
CODEC_PARAMS="++pipeline.codec.decode_only=True" 
EXPERIMENT=""
DEVICE="cpu"
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
        sbatch --mem=64G -c 2 --reservation=deepvideo2 --job-name=oiv6_seg_decode $1
        # grid batch --job-name=oiv6_seg_decode --reservation=deepvideo2 --nodelist=kopspgd16p --cpus 2 -- bash $1
    }
    export -f run_scripts
else
    run_scripts () {
        bash $1
    }
    export -f run_scripts
fi


# DECODE + NN_PART2
for QP_DIR in $( find ${BITSTREAM_DIR} -type f -name "mpeg-oiv6-segmentation*.bin"  | xargs dirname | sort | uniq );
do
    echo $QP_DIR
    QP=$(echo "$QP_DIR" | grep -oP '(?<=qp)[^_]*(?=_qpdensity)' | tail -n 1)
    echo RUN: ${RUN}, Input Dir: ${INPUT_DIR}, Bitstream Dir: ${BITSTREAM_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}, CODEC_PARAMS: ${CODEC_PARAMS}
    run_scripts "../mpeg_cfp_oiv6_segmentation.sh ${INPUT_DIR} ${BITSTREAM_DIR} '${EXPERIMENT}' ${DEVICE} ${QP} ${CODEC_PARAMS}"
done


# GENERATE CSV
if [[ ${RUN} == "gnu_parallel" ]]; then
    sem --wait
    bash gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
elif [[ ${RUN} == "slurm" ]]; then
    sbatch --dependency=singleton --job-name=oiv6_seg_decode  gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
else
    bash gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
fi
