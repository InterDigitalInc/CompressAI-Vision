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
if [[ "*${CONF_NAME}*" == "remote" ]]; then
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
echo "Pipeline Type:      " ${CMD}
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
        ++pipeline.type=image \
        ++paths._run_root=${OUTPUT_DIR} \
        ++vision_model.arch=${NETWORK_MODEL} \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=MPEGOIV6 \
        ++dataset.config.root=${DATASET_SRC} \
        ++dataset.config.annotation_file=annotations/${SEQ}-coco.json \
        ++dataset.config.dataset_name=${SEQ} \
        ++evaluator.type=OIC-EVAL \
        ++codec.experiment=${EXPERIMENT} \
        codec=hm.yaml \
        ++codec.encoder_config.intra_period=${INTRA_PERIOD} \
        ++codec.encoder_config.parallel_encoding=False \
        ++codec.encoder_config.qp=${QP} \
	++codec.codec_paths.encoder_exe=${INNER_CODEC_PATH}'/bin/TAppEncoderStatic'  \
        ++codec.codec_paths.decoder_exe=${INNER_CODEC_PATH}'/bin/TAppDecoderStatic' \
        ++codec.codec_paths.parcat_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
	++codec.codec_paths.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_intra_main10.cfg' \
        ++codec.eval_encode='bpp' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
