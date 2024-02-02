#!/usr/bin/env bash
set -eu
export DNNL_MAX_CPU_ISA=AVX2

FCM_TESTDATA=$1
INNER_CODEC_PATH=$2
OUTPUT_DIR=$3
EXPERIMENT=$4
DEVICE=$5
QP=$6
SEQ=$7
PIPELINE_PARAMS=$8
CONF_NAME=$9

DATASET_SRC="${FCM_TESTDATA}/mpeg-oiv6"

CMD="compressai-vision-eval"

declare -A network_model
declare -A task_type

network_model["mpeg-oiv6-detection"]="faster_rcnn_X_101_32x8d_FPN_3x"
task_type["mpeg-oiv6-detection"]="obj"

network_model["mpeg-oiv6-segmentation"]="mask_rcnn_X_101_32x8d_FPN_3x"
task_type["mpeg_oiv6-segmentation"]="seg"


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
        "--config-name=eval_remote_inference_example.yaml" \
        "codec=vtm.yaml" \
	++codec.enc_configs.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_intra_vtm.cfg' \
        ++codec.enc_configs.frame_rate=${FRAME_RATE} \
        ++codec.enc_configs.intra_period=${INTRA_PERIOD} \
        ++codec.enc_configs.parallel_encoding=False \
        ++codec.enc_configs.qp=${QP} \
        "++codec.codec_paths.encoder_exe=/los/home/racapef/vvc/vtm-12.0/bin/EncoderAppStatic" \
        "++codec.codec_paths.decoder_exe=/los/home/racapef/vvc/vtm-12.0/bin/DecoderAppStatic" \
        "++codec.codec_paths.parcat_exe=/los/home/racapef/vvc/vtm-12.0/bin/parcatStatic" \
        "++codec.codec_paths.cfg_file=/los/home/racapef/vvc/vtm-12.0/cfg/encoder_lowdelay_P_vtm.cfg" \
	++codec.tools.inner_codec.enc_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.tools.inner_codec.dec_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.tools.inner_codec.merge_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
        ++codec.eval_encode='bpp' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
