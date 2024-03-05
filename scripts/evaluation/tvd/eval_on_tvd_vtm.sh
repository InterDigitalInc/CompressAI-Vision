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
export DEVICE=${DEVICE}

DATASET_SRC="${FCM_TESTDATA}/tvd_tracking"

CMD="compressai-split-inference"
if [[ "*${CONF_NAME}*" == "remote" ]]; then
  CMD="compressai-remote-inference"
fi

declare -A intra_period_dict
declare -A fr_dict

intra_period_dict["TVD-01"]=64
fr_dict["TVD-01"]=50

intra_period_dict["TVD-02"]=64
fr_dict["TVD-02"]=50

intra_period_dict["TVD-03"]=64
fr_dict["TVD-03"]=50

INTRA_PERIOD=${intra_period_dict[${SEQ}]}
FRAME_RATE=${fr_dict[${SEQ}]}

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

${CMD} --config-name=${CONF_NAME}.yaml \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
	++vision_model.arch=jde_1088x608 \
        ++vision_model.jde_1088x608.splits="[36, 61, 74]" \
        ++dataset.type=TrackingDataset \
        ++dataset.datacatalog=MPEGTVDTRACKING \
	++dataset.settings.patch_size="[608, 1088]" \
        ++dataset.config.root=${DATASET_SRC}/${SEQ} \
        ++dataset.config.imgs_folder=img1 \
       	++dataset.config.annotation_file=gt/gt.txt \
        ++dataset.config.dataset_name=mpeg-${SEQ} \
        ++evaluator.type=MOT-TVD-EVAL \
        ++codec.experiment=${EXPERIMENT} \
	codec=vtm.yaml \
        ++codec.encoder_config.intra_period=${INTRA_PERIOD} \
        ++codec.encoder_config.parallel_encoding=True \
        ++codec.encoder_config.qp=${QP} \
        ++codec.codec_paths.encoder_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.codec_paths.decoder_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.codec_paths.parcat_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
        ++codec.codec_paths.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_lowdelay_vtm.cfg' \
        ++codec.eval_encode='bitrate' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
        ${PIPELINE_PARAMS} \
        