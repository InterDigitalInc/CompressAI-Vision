#! /usr/bin/env bash

# grid batch --cpus=18 --job-name=tvd -o tvd.out -- bash run_tvd.sh

# change the paths accrodingly
#####################################################
INPUT_DIR="/data/datasets/MPEG-FCVCM/vcm_testdata"
OUTPUT_DIR="./cfp_run"
EXPERIMENT="_tvd_cfp_test"
DEVICE="cpu"
#####################################################
MAX_PARALLEL=18 # total number of jobs = 18

QPS=`echo "3 2 1 -1 -2 -3"`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

parallel_run () {
	sem -j $MAX_PARALLEL bash "$1"
}

export -f parallel_run

# TVD - Object Tracking with JDE
for SEQ in \
            'TVD-01' \
            'TVD-02' \
            'TVD-03'
do
    for QP in ${QPS}
    do
        echo Input Dir: ${INPUT_DIR}, Output Dir: ${OUTPUT_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}, SEQ Name: ${SEQ}
        parallel_run "${SCRIPT_DIR}/../mpeg_cfp_tvd.sh ${INPUT_DIR} ${OUTPUT_DIR} ${EXPERIMENT} ${DEVICE} ${QP} ${SEQ}"
    done
done

sem --wait
python ${SCRIPT_DIR}/../../utils/mpeg_template_format.py --dataset TVD --result_path ${OUTPUT_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGTVDTRACKING/

