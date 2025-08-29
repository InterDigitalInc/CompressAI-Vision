#!/usr/bin/env bash
#
# This runs the evaluation of original models, whitout compression 
# make sure you sourced the virtual environment that contains up-to-date installed compressai-vision
# see provided installation scripts
set -eu

ENTRY_CMD=$1
TESTDATA_DIR=$2
DEVICE=$3

# List of entry cmds 
CMD_OPTS=("compressai-split-inference", "compressai-remote-inference")

if [[ "${CMD_OPTS[@]}" =~ ${ENTRY_CMD} ]]; then
    echo "Run ${ENTRY_CMD} ........"
else
    echo : "${ENTRY_CMD} does not exist in the options."
    echo : "Please choose one out of these options: ${CMD_OPTS[@]}"
    exit 1
fi

declare -A configs

configs["compressai-split-inference"]="eval_split_inference_example"
configs["compressai-remote-inference"]="eval_remote_inference_example"

CONF_NAME=${configs[${ENTRY_CMD}]}

if [ $# == 2 ]; then
    TESTDATA_DIR=$2
fi
if [ ! -d "${TESTDATA_DIR}" ]; then
    echo "${TESTDATA_DIR} does not exist, please select dataset folder, e.g.
    $ bash default_vision_performances.sh [etnry_cmd] [/path/to/dataset]"
    exit
fi

MPEG_SAM="${TESTDATA_DIR}/sam_test"



# MPEGOIV6 - Small test
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             ++pipeline.type=image \
             ++vision_model.arch=sam_vit_h_4b8939 \
             ++dataset.type=SamDataset \
             ++dataset.datacatalog=MPEGSAM \
             ++dataset.config.root=${MPEG_SAM} \
             ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco_2images.json \
             ++dataset.config.dataset_name=mpeg-oiv6-sam \
             ++evaluator.type=OIC-EVAL \
             ++evaluator.overwrite_results=True \
             ++codec.encoder_config.parallel_encoding=False \
             ++pipeline.nn_task_part1.load_features=False \
             ++pipeline.nn_task_part1.dump_features=False \
             ++pipeline.nn_task_part2.dump_features=False \
             ++misc.device.nn_parts=${DEVICE} \
             ++codec.eval_encode='bpp' \

             # ++pipeline.conformance.save_conformance_files=True \
             # ++pipeline.conformance.subsample_ratio=9 \


