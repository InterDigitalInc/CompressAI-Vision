#!/usr/bin/env bash
set -eu

VCM_TESTDATA=$1
OUTPUT_DIR=$2
EXPERIMENT=$3
DEVICE=$4
QP=$5
SEQ=$6
CODEC_PARAMS=$7

export DNNL_MAX_CPU_ISA=AVX2
export DEVICE=$device


SFU_HW_SRC="${VCM_TESTDATA}/SFU_HW_Obj"

CONF_NAME="eval_vtm"

CMD="compressai-vision-eval"


declare -A intra_period_dict
declare -A fr_dict

intra_period_dict["Traffic_2560x1600_30_val"]=32
fr_dict["Traffic_2560x1600_30_val"]=30

intra_period_dict["Kimono_1920x1080_24_val"]=32
fr_dict["Kimono_1920x1080_24_val"]=24

intra_period_dict["ParkScene_1920x1080_24_val"]=32
fr_dict["ParkScene_1920x1080_24_val"]=24

intra_period_dict["Cactus_1920x1080_50_val"]=64
fr_dict["Cactus_1920x1080_50_val"]=50

intra_period_dict["BasketballDrive_1920x1080_50_val"]=64
fr_dict["BasketballDrive_1920x1080_50_val"]=50

intra_period_dict["BasketballDrill_832x480_50_val"]=64
fr_dict["BasketballDrill_832x480_50_val"]=50

intra_period_dict["BQTerrace_1920x1080_60_val"]=64
fr_dict["BQTerrace_1920x1080_60_val"]=60

intra_period_dict["BQSquare_416x240_60_val"]=64
fr_dict["BQSquare_416x240_60_val"]=60

intra_period_dict["PartyScene_832x480_50_val"]=64
fr_dict["PartyScene_832x480_50_val"]=50

intra_period_dict["RaceHorses_832x480_30_val"]=32
fr_dict["RaceHorses_832x480_30_val"]=30

intra_period_dict["RaceHorses_416x240_30_val"]=32
fr_dict["RaceHorses_416x240_30_val"]=30

intra_period_dict["BlowingBubbles_416x240_50_val"]=64
fr_dict["BlowingBubbles_416x240_50_val"]=50

intra_period_dict["BasketballPass_416x240_50_val"]=64
fr_dict["BasketballPass_416x240_50_val"]=50

intra_period_dict["BQMall_832x480_60_val"]=64
fr_dict["BQMall_832x480_60_val"]=60

INTRA_PERIOD=${intra_period_dict[${SEQ}]}
FRAME_RATE=${fr_dict[${SEQ}]}

echo ${VCM_TESTDATA}, ${OUTPUT_DIR}, ${EXPERIMENT}, ${DEVICE}, ${QP}, ${SEQ}, ${CODEC_PARAMS}, ${INTRA_PERIOD}, ${FRAME_RATE}

${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
        ++codec.codec_paths.cfg_file='/pa/projects/etl-vcm/fcvcm/vtm-12.0/cfg/encoder_randomaccess_vtm.cfg' \
        ++codec.encoder_config.frame_rate=${FRAME_RATE} \
        ++codec.encoder_config.intra_period=${INTRA_PERIOD} \
        ++codec.encoder_config.qp=${QP} \
        ++codec.eval_encode='bitrate' \
        ++codec.experiment=${EXPERIMENT} \
        ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=SFUHW \
        ++dataset.config.root=${SFU_HW_SRC}/${SEQ} \
        ++dataset.config.annotation_file=annotations/${SEQ}.json \
        ++dataset.config.dataset_name=sfu-hw-${SEQ} \
        ++evaluator.type=COCO-EVAL \
        ++misc.device="${DEVICE}"
