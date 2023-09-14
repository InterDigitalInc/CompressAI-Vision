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

HIEVE_SRC="${VCM_TESTDATA}/HiEve_pngs"

CONF_NAME="eval_cfp_codec"

CMD="compressai-vision-eval"

${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=90 \
        ++codec.encoder_config.qp=${QP} \
        ++codec.eval_encode='bitrate' \
        ++codec.experiment=${EXPERIMENT} \
        ++vision_model.arch=jde_1088x608 \
        ++vision_model.jde_1088x608.splits="[75, 90, 105]" \
        ++dataset.type=TrackingDataset \
        ++dataset.settings.patch_size="[608, 1088]" \
        ++dataset.datacatalog=MPEGHIEVE \
        ++dataset.config.root=${HIEVE_SRC}/${SEQ} \
        ++dataset.config.imgs_folder=img1 \
        ++dataset.config.annotation_file=gt/gt.txt \
        ++dataset.config.dataset_name=mpeg-hieve-${SEQ} \
        ++evaluator.type=MOT-HIEVE-EVAL \
        ++misc.device="${DEVICE}"
