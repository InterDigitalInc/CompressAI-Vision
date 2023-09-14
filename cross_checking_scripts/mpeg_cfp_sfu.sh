#!/usr/bin/env bash
set -eu
export DNNL_MAX_CPU_ISA=AVX2

VCM_TESTDATA=$1
OUTPUT_DIR=$2
EXPERIMENT=$3
DEVICE=$4
qp=$5
SEQ=$6
CODEC_PARAMS=$7

echo ${VCM_TESTDATA}, ${OUTPUT_DIR}, ${EXPERIMENT}, ${DEVICE}, ${QP}, ${SEQ}, ${CODEC_PARAMS}

SFU_HW_SRC="${VCM_TESTDATA}/SFU_HW_Obj"

CONF_NAME="eval_cfp_codec"

CMD="compressai-vision-eval"

${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
        ++codec.encoder_config.qp=${qp} \
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
