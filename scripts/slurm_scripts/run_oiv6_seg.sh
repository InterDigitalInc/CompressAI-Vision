#! /usr/bin/env bash

RUN="slurm" # "gnu_parallel" or "sequential" or "slurm"

INPUT_DIR="/mnt/koko-storage/mlab/eimran/vcm_testdata/" # needed for NN_PART2
BITSTREAM_DIR="/mnt/koko-storage/mlab/eimran/outputs/"

#################################################################
EXPERIMENT="_cfp_adaptive_clustering"
DEVICE="cpu"
config_name="eval_cfp_codec"
QPS=`echo "9 10 11 12 13 14"`
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
        sbatch -p longq7 --mem=32G -c 4 --job-name=oiv6_seg $1
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
    run_scripts "../gridfiles/mpeg_cfp_oiv6_segmentation.sh ${INPUT_DIR} ${BITSTREAM_DIR} ${EXPERIMENT} ${DEVICE} ${QP} ${config_name}"
done


# GENERATE CSV
if [[ ${RUN} == "gnu_parallel" ]]; then
    sem --wait
    bash gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
elif [[ ${RUN} == "slurm" ]]; then
    sbatch --dependency=singleton --job-name=oiv6_seg  gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
else
    bash gen_csv.sh OIV6 ${BITSTREAM_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
fi
