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

DATASET_SRC="${FCM_TESTDATA}/SFU_HW_Obj"

CMD="compressai-split-inference"
if [[ "*${CONF_NAME}*" == "remote" ]]; then
  CMD="compressai-remote-inference"
fi

declare -A intra_period_dict
declare -A fr_dict

intra_period_dict["Traffic_2560x1600_30_val"]=32
fr_dict["Traffic_2560x1600_30_val"]=30

intra_period_dict["Kimono_1920x1080_24_val"]=32
fr_dict["Kimono_1920x1080_24_val"]=24

intra_period_dict["ParkScene_1920x1080_24_val"]=32
fr_dict["ParkScene_1920x1080_24_val"]=24

intra_period_dict["Cactus_1920x1080_50_val"]=64
fr_dict["Cactus_1920x1080_50_val"]=50

intra_period_dict["BasketballDrive_1920x1080_50_val"]=64
fr_dict["BasketballDrive_1920x1080_50_val"]=50

intra_period_dict["BasketballDrill_832x480_50_val"]=64
fr_dict["BasketballDrill_832x480_50_val"]=50

intra_period_dict["BQTerrace_1920x1080_60_val"]=64
fr_dict["BQTerrace_1920x1080_60_val"]=60

intra_period_dict["BQSquare_416x240_60_val"]=64
fr_dict["BQSquare_416x240_60_val"]=60

intra_period_dict["PartyScene_832x480_50_val"]=64
fr_dict["PartyScene_832x480_50_val"]=50

intra_period_dict["RaceHorses_832x480_30_val"]=32
fr_dict["RaceHorses_832x480_30_val"]=30

intra_period_dict["RaceHorses_416x240_30_val"]=32
fr_dict["RaceHorses_416x240_30_val"]=30

intra_period_dict["BlowingBubbles_416x240_50_val"]=64
fr_dict["BlowingBubbles_416x240_50_val"]=50

intra_period_dict["BasketballPass_416x240_50_val"]=64
fr_dict["BasketballPass_416x240_50_val"]=50

intra_period_dict["BQMall_832x480_60_val"]=64
fr_dict["BQMall_832x480_60_val"]=60

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
	++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=SFUHW \
        ++dataset.config.root=${DATASET_SRC}/${SEQ} \
        ++dataset.config.annotation_file=annotations/${SEQ}.json \
        ++dataset.config.dataset_name=sfu-hw-${SEQ} \
        ++evaluator.type=COCO-EVAL \
	++evaluator.eval_criteria=AP50 \
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
