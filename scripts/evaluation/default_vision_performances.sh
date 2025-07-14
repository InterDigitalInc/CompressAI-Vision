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

MPEG_OIV6_SRC="${TESTDATA_DIR}/mpeg-oiv6"
SFU_HW_SRC="${TESTDATA_DIR}/SFU_HW_Obj"
HIEVE_SRC="${TESTDATA_DIR}/HiEve_pngs"
TVD_SRC="${TESTDATA_DIR}/tvd_tracking"
PANDASET_SRC="${TESTDATA_DIR}/PandaSet"

# MPEGOIV6 - Detection with Faster RCNN
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             ++pipeline.type=image \
             ++pipeline.conformance.save_conformance_files=True \
             ++pipeline.conformance.subsample_ratio=9 \
             ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
             ++dataset.type=Detectron2Dataset \
             ++dataset.datacatalog=MPEGOIV6 \
             ++dataset.config.root=${MPEG_OIV6_SRC} \
             ++dataset.config.annotation_file=annotations/mpeg-oiv6-detection-coco.json \
             ++dataset.config.dataset_name=mpeg-oiv6-detection \
             ++evaluator.type=OIC-EVAL \
             ++pipeline.nn_task_part1.load_features=False \
             ++pipeline.nn_task_part1.dump_features=False \
             ++pipeline.nn_task_part2.dump_features=False \
             ++misc.device.nn_parts=${DEVICE}

# MPEGOIV6 - Segmentation with Mask RCNN
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             ++pipeline.type=image \
             ++pipeline.conformance.save_conformance_files=True \
             ++pipeline.conformance.subsample_ratio=9 \
             ++vision_model.arch=mask_rcnn_X_101_32x8d_FPN_3x \
             ++dataset.type=Detectron2Dataset \
             ++dataset.datacatalog=MPEGOIV6 \
             ++dataset.config.root=${MPEG_OIV6_SRC} \
             ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco.json \
             ++dataset.config.dataset_name=mpeg-oiv6-segmentation \
             ++evaluator.type=OIC-EVAL \
             ++pipeline.nn_task_part1.load_features=False \
             ++pipeline.nn_task_part1.dump_features=False \
             ++pipeline.nn_task_part2.dump_features=False \
             ++misc.device.nn_parts=${DEVICE}

# SFU - Detection with Faster RCNN
for SEQ in \
            'Traffic_2560x1600_30_val' \
            'Kimono_1920x1080_24_val' \
            'ParkScene_1920x1080_24_val' \
            'Cactus_1920x1080_50_val' \
            'BasketballDrive_1920x1080_50_val' \
            'BQTerrace_1920x1080_60_val' \
            'BasketballDrill_832x480_50_val' \
            'BQMall_832x480_60_val' \
            'PartyScene_832x480_50_val' \
            'RaceHorses_832x480_30_val' \
            'BasketballPass_416x240_50_val' \
            'BQSquare_416x240_60_val' \
            'BlowingBubbles_416x240_50_val' \
            'RaceHorses_416x240_30_val'
do
    ${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
                 ++pipeline.type=video \
                 ++pipeline.conformance.save_conformance_files=True \
                 ++pipeline.conformance.subsample_ratio=9 \
                 ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
                 ++dataset.type=Detectron2Dataset \
                 ++dataset.datacatalog=SFUHW \
                 ++dataset.config.root=${SFU_HW_SRC}/${SEQ} \
                 ++dataset.config.annotation_file=annotations/${SEQ}.json \
                 ++dataset.config.dataset_name=${SEQ} \
                 ++evaluator.type=COCO-EVAL \
                 ++pipeline.nn_task_part1.load_features=False \
                 ++pipeline.nn_task_part1.dump_features=False \
                 ++pipeline.nn_task_part2.dump_features=False \
                 ++misc.device.nn_parts=${DEVICE}
done

# TVD - Object Tracking with JDE
for SEQ in \
            'TVD-01' \
            'TVD-02' \
            'TVD-03'
do
    ${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
                 ++pipeline.type=video \
             	 ++pipeline.conformance.save_conformance_files=True \
             	 ++pipeline.conformance.subsample_ratio=9 \
                 ++vision_model.arch=jde_1088x608 \
                 ++vision_model.jde_1088x608.splits="[36, 61, 74]" \
                 ++dataset.type=TrackingDataset \
                 ++dataset.settings.patch_size="[608, 1088]" \
    	         ++dataset.datacatalog=MPEGTVDTRACKING \
                 ++dataset.config.root=${TVD_SRC}/${SEQ} \
                 ++dataset.config.imgs_folder=img1 \
                 ++dataset.config.annotation_file=gt/gt.txt \
                 ++dataset.config.dataset_name=${SEQ} \
                 ++evaluator.type=MOT-TVD-EVAL \
                 ++pipeline.nn_task_part1.load_features=False \
                 ++pipeline.nn_task_part1.dump_features=False \
                 ++pipeline.nn_task_part2.dump_features=False \
                 ++misc.device.nn_parts=${DEVICE}
done

# HIEVE - Object Tracking with JDE
for SEQ in \
            '13' \
            '16' \
            '2' \
            '17' \
            '18'
do
    ${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
                 ++pipeline.type=video \
                 ++pipeline.conformance.save_conformance_files=True \
                 ++pipeline.conformance.subsample_ratio=90 \
                 ++vision_model.arch=jde_1088x608 \
                 ++vision_model.jde_1088x608.splits="[105, 90, 75]" \
                 ++dataset.type=TrackingDataset \
                 ++dataset.settings.patch_size="[608, 1088]" \
                 ++dataset.datacatalog=MPEGHIEVE \
                 ++dataset.config.root=${HIEVE_SRC}/${SEQ} \
                 ++dataset.config.imgs_folder=img1 \
                 ++dataset.config.annotation_file=gt/gt.txt \
                 ++dataset.config.dataset_name=${SEQ} \
                 ++evaluator.type=MOT-HIEVE-EVAL \
                 ++pipeline.nn_task_part1.load_features=False \
                 ++pipeline.nn_task_part1.dump_features=False \
                 ++pipeline.nn_task_part2.dump_features=False \
                 ++misc.device.nn_parts=${DEVICE}
done

# PANDASET - Semantic Segmentation with Pandaset
for SEQ in \
            '003' \
            '011' \
            '016' \
            '017' \
            '021' \
            '023' \
            '027' \
            '029' \
            '030' \
            '033' \
            '035' \
            '037' \
            '039' \
            '043' \
            '053' \
            '056' \
            '057' \
            '058' \
            '069' \
            '070' \
            '072' \
            '073' \
            '077' \
            '088' \
            '089' \
            '090' \
            '095' \
            '097' \
            '109' \
            '112' \
            '113' \
            '115' \
            '117' \
            '119' \
            '122' \
            '124'
do
    ${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
                 ++pipeline.type=video \
                 ++pipeline.conformance.save_conformance_files=True \
                 ++pipeline.conformance.subsample_ratio=9 \
                 ++vision_model.arch=panoptic_rcnn_R_101_FPN_3x \
                 ++dataset.type=Detectron2Dataset \
                 ++dataset.datacatalog=PANDASET \
                 ++dataset.config.root=${PANDASET_SRC}/${SEQ} \
                 ++dataset.config.imgs_folder=camera/front_camera \
                 ++dataset.config.ext=jpg \
                 ++dataset.config.annotation_file=annotations/${SEQ}.npz \
                 ++dataset.config.dataset_name=pandaset-${SEQ} \
                 ++evaluator.type=SEMANTICSEG-EVAL \
                 ++pipeline.nn_task_part1.load_features=False \
                 ++pipeline.nn_task_part1.dump_features=False \
                 ++pipeline.nn_task_part2.dump_features=False \
                 ++misc.device.nn_parts=${DEVICE}
done