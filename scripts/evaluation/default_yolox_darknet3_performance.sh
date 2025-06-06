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

# /path/to/COCODataset/
COCO_2017_VAL_SRC="${TESTDATA_DIR}/coco2017"

# COCO 2017 Val - Detection with YOLOX-Darknet53

# option for split points "l13" or "l37"
# ++vision_model.yolox_darknet53.splits="l37" \ 
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             ++pipeline.type=image \
             ++pipeline.conformance.save_conformance_files=False \
             ++vision_model.arch=yolox_darknet53 \
             ++dataset.type=YOLOXDataset \
             ++dataset.settings.patch_size="[640, 640]" \
             ++dataset.datacatalog=COCO \
             ++dataset.config.root=${COCO_2017_VAL_SRC} \
             ++dataset.config.annotation_file=annotations/instances_val2017.json \
             ++dataset.config.imgs_folder=val2017 \
             ++dataset.config.dataset_name=coco2017val \
             ++evaluator.type=YOLOX-COCO-EVAL \
             ++evaluator.overwrite_results=True \
             ++pipeline.nn_task_part1.load_features=False \
             ++pipeline.nn_task_part1.dump_features=False \
             ++pipeline.nn_task_part2.dump_features=False \
             ++codec.eval_encode=bpp \
             ++misc.device.nn_parts=${DEVICE}
  