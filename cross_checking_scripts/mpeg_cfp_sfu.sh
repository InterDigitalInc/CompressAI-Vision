#!/usr/bin/env bash
set -eu

VCM_TESTDATA=$1
OUTPUT_DIR=$2
EXPERIMENT=$3
DEVICE=$4
QP=$5
SEQ=$6

SFU_HW_SRC="${VCM_TESTDATA}/SFU_HW_Obj"

CONF_NAME="eval_cfp_codec"
# CONF_NAME="eval_vtm"
# CONF_NAME="eval_ffmpeg"

CODEC_PARAMS=""
# e.g.
# CODEC_PARAMS="++codec.type=x265"


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CMD="compressai-vision-eval"


${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
            ++pipeline.type=video \
            ++paths._runs_root=${OUTPUT_DIR} \
            ++pipeline.conformance.save_conformance_files=True \
            ++pipeline.conformance.subsample_ratio=9 \
            ++codec.encoder_config.n_cluster='{p2: 128, p3: 128, p4: 150, p5: 180}'  \
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