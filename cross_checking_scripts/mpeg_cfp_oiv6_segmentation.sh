#!/usr/bin/env bash
set -eu

VCM_TESTDATA=$1
OUTPUT_DIR=$2
EXPERIMENT=$3
DEVICE=$4
qp=$5

MPEG_OIV6_SRC="${VCM_TESTDATA}/mpeg-oiv6"

CONF_NAME="eval_cfp_codec"
# CONF_NAME="eval_vtm"
# CONF_NAME="eval_ffmpeg"

CODEC_PARAMS=""
# e.g.
# CODEC_PARAMS="++codec.type=x265"

CMD="compressai-vision-eval"

echo "running segmentation task with qp=${qp}"  
${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++paths._runs_root="${OUTPUT_DIR}" \
        ++pipeline.type=image \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
        ++codec.encoder_config.feature_channel_suppression.manual_cluster=False \
        ++codec.encoder_config.feature_channel_suppression.min_nb_channels_for_group=3 \
        ++codec.encoder_config.feature_channel_suppression.xy_margin=0.15 \
        ++codec.encoder_config.feature_channel_suppression.coverage_thres=0.7 \
        ++codec.encoder_config.feature_channel_suppression.coverage_decay=0.98 \
        ++codec.encoder_config.feature_channel_suppression.downscale=True \
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
