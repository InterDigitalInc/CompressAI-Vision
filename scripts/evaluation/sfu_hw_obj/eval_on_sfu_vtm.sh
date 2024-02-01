#!/usr/bin/env bash
set -eu

FCM_TESTDATA=$1
INNER_CODEC_PATH=$2
OUTPUT_DIR=$3
EXPERIMENT=$4
DEVICE=$5
QP=$6
SEQ=$7
PIPELINE_PARAMS=$8
CONF_NAME=$9

export DNNL_MAX_CPU_ISA=AVX2
export DEVICE=$device


DATASET_SRC="${FCM_TESTDATA}/mpeg-oiv6"

CMD="compressai-split-inference"
if [[ $remote == *${CONF_NAME}* ]]; then
  CMD="compressai-remote-inference"
fi

declare -A network_model
declare -A task_type

network_model["mpeg-oiv6-detection"]="faster_rcnn_X_101_32x8d_FPN_3x"
task_type["mpeg-oiv6-detection"]="obj"

network_model["mpeg-oiv6-segmentation"]="mask_rcnn_X_101_32x8d_FPN_3x"
task_type["mpeg-oiv6-segmentation"]="seg"


NETWORK_MODEL=${network_model[${SEQ}]}
TASK_TYPE=${task_type[${SEQ}]}
INTRA_PERIOD=1
FRAME_RATE=1

echo "============================== RUNNING COMPRESSAI-VISION EVAL== =================================="
echo "Datatset location:  " ${FCM_TESTDATA}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "fctm"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Input sequence:     " ${SEQ}
echo "Seq. Framerate:     " ${FRAME_RATE}
echo "QP for Inner Codec: " ${QP}
echo "Intra Period for Inner Codec: "${INTRA_PERIOD}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="
 
echo "============================== RUNNING COMPRESSAI-VISION EVAL== =================================="
echo "Datatset location:  " ${FCM_TESTDATA}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "fctm"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Input sequence:     " ${SEQ}
echo "Seq. Framerate:     " ${FRAME_RATE}
echo "QP for Inner Codec: " ${QP}
echo "Intra Period for Inner Codec: "${INTRA_PERIOD}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="
 
${CMD} --config-name=${CONF_NAME}.yaml ${PIPELINE_PARAMS} \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
	++codec.codec_paths.encoder_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.codec_paths.encoder_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.codec_paths.encoder_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
	++codec.enc_configs.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_lowdelay_vtm.cfg' \
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
        ++codec.verbosity=1 \
	++codec.device=${DEVICE} \
        ++misc.device="${DEVICE}"
