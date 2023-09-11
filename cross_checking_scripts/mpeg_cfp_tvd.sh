#!/usr/bin/env bash
set -eu

VCM_TESTDATA=$1
OUTPUT_DIR=$2
EXPERIMENT=$3
DEVICE=$4
QP=$5
SEQ=$6

TVD_SRC="${VCM_TESTDATA}/tvd_tracking"

CONF_NAME="eval_cfp_codec"
# CONF_NAME="eval_vtm"
# CONF_NAME="eval_ffmpeg"

CODEC_PARAMS=""
# e.g.
# CODEC_PARAMS="++codec.type=x265"

CMD="compressai-vision-eval"

${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++pipeline.type=video \
        ++paths._runs_root=${OUTPUT_DIR} \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=90 \
        ++codec.encoder_config.feature_channel_suppression.manual_cluster=True \
        ++codec.encoder_config.feature_channel_suppression.n_clusters='{36: 128, 61: 128, 74: 128}' \
        ++codec.encoder_config.feature_channel_suppression.downscale=True \
        ++codec.encoder_config.qp=${QP} \
        ++codec.eval_encode='bitrate' \
        ++codec.experiment=${EXPERIMENT} \
        ++vision_model.arch=jde_1088x608 \
        ++vision_model.jde_1088x608.splits="[36, 61, 74]" \
        ++dataset.type=TrackingDataset \
        ++dataset.settings.patch_size="[608, 1088]" \
        ++dataset.datacatalog=MPEGTVDTRACKING \
        ++dataset.config.root=${TVD_SRC}/${SEQ} \
        ++dataset.config.imgs_folder=img1 \
        ++dataset.config.annotation_file=gt/gt.txt \
        ++dataset.config.dataset_name=mpeg-tracking-${SEQ} \
        ++evaluator.type=MOT-TVD-EVAL \
        ++misc.device="${DEVICE}"
