#!/usr/bin/env bash
set -eu
export DNNL_MAX_CPU_ISA=AVX2

VCM_TESTDATA=$1
OUTPUT_DIR=$2
EXPERIMENT=$3
DEVICE=$4
qp=$5
CODEC_PARAMS=$6

echo ${VCM_TESTDATA}, ${OUTPUT_DIR}, ${EXPERIMENT}, ${DEVICE}, ${qp}, ${CODEC_PARAMS}

MPEG_OIV6_SRC="${VCM_TESTDATA}/mpeg-oiv6"

CONF_NAME="eval_cfp_codec"

CMD="compressai-vision-eval"

echo "running segmentation task with qp=${qp}"  
${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++paths._runs_root="${OUTPUT_DIR}" \
        ++pipeline.type=image \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
        ++codec.encoder_config.feature_channel_suppression.manual_cluster=False \
        ++codec.encoder_config.feature_channel_suppression.min_nb_channels_for_group=3 \
        ++codec.encoder_config.feature_channel_suppression.downscale=True \
        ++codec.encoder_config.feature_channel_suppression.supression_measure='rpn' \
        ++codec.encoder_config.feature_channel_suppression.rpn.xy_margin=0.10 \
        ++codec.encoder_config.feature_channel_suppression.rpn.xy_margin_decay=0.01 \
        ++codec.encoder_config.feature_channel_suppression.rpn.coverage_thres=0.75 \
        ++codec.encoder_config.feature_channel_suppression.rpn.coverage_decay=0.98 \
        ++codec.encoder_config.qp=${qp} \
        ++codec.eval_encode='bpp' \
        ++codec.experiment=${EXPERIMENT} \
        ++vision_model.arch=mask_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=MPEGOIV6 \
        ++dataset.config.root=${MPEG_OIV6_SRC} \
        ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco.json \
        ++dataset.config.dataset_name=mpeg-oiv6-segmentation \
        ++evaluator.type=OIC-EVAL \
        ++misc.device="${DEVICE}"
