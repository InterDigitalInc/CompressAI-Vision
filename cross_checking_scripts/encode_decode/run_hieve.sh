#! /usr/bin/env bash

# grid batch --nodelist=kopspgd16p --cpus=12 --job-name=hieve -o hieve.out -- bash run_hieve.sh

# change the paths accrodingly
#####################################################
INPUT_DIR="/data/datasets/MPEG-FCVCM/vcm_testdata"
OUTPUT_DIR="./cfp_run"
EXPERIMENT="_hieve_cfp_test"
DEVICE="cpu"
#####################################################
MAX_PARALLEL=2 # total number of jobs = 25

QPS=`echo "3 2 1 -1 -2 -3"`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

parallel_run () {
	sem -j $MAX_PARALLEL bash "$1"
}

export -f parallel_run

# HIEVE - Object Tracking with JDE
for SEQ in \
            '13' \
            '16' \
            '2' \
            '17' \
            '18'
do
    for QP in ${QPS}
    do
        echo Input Dir: ${INPUT_DIR}, Output Dir: ${OUTPUT_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}, SEQ Name: ${SEQ}
        parallel_run "${SCRIPT_DIR}/../mpeg_cfp_hieve.sh ${INPUT_DIR} ${OUTPUT_DIR} ${EXPERIMENT} ${DEVICE} ${QP} ${SEQ}"
    done
done

sem --wait
python ${SCRIPT_DIR}/../../utils/mpeg_template_format.py --dataset HIEVE --result_path ${OUTPUT_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGHIEVE/

