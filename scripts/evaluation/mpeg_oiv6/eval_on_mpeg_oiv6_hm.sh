#!/usr/bin/env bash
set -eu

FCM_TESTDATA=""
INNER_CODEC_PATH=""
OUTPUT_DIR=""
EXPERIMENT="test"
DEVICE="cpu"
QP=42
SEQ=""
PIPELINE_PARAMS=""
PIPELINE="split"

#parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_
Runs an evaluation pipeline (split or remote) using HM as codec

RUN OPTIONS:
                [-t|--testdata) path to the test dataset, e.g. path/to/fcm_testdata/, default=""]
                [-p|--pipeline) type of pipeline, split/remote, default="split"]
                [-i|--inner_codec) path to core codec, e.g. /path/to/HM_repo (that contains subfolders bin/ cfg/ ...), default=""]
                [-o|--output_dir) root of output folder for logs and artifacts, default=""]
                [-e|--exp_name) name of the experiments to locate/label outputs, default="test"]
                [-d|--device) cuda or cpu, default="cpu"]
                [-q|--qp) quality level, depends on the inner codec, default=42]
                [-s|--seq_name) sequence name as used in testdata root folder. E.g., "Traffic_2560x1600_30_val" in sfu_hw_obj, default="42"]
                [-x|--extra_params) additional parameters to override default configs (pipeline/codec/evaluation...), default=""]
EXAMPLE         [bash eval_on_mpeg_oiv6_hm.sh -t /path/to/testdata -p split -i /path/to/VTM_repo -d cpu -q 32 -s Traffic_2560x1600_30_val]
_EOF_
            exit;
            ;;
        -t|--testdata) shift; FCM_TESTDATA="$1"; shift; ;;
        -p|--pipeline) shift; PIPELINE="$1"; shift; ;;
        -i|--inner_codec) shift; INNER_CODEC_PATH="$1"; shift; ;;
        -o|--output_dir) shift; OUTPUT_DIR="$1"; shift; ;;
        -e|--exp_name) shift; EXPERIMENT="$1"; shift; ;;
        -d|--device) shift; DEVICE="$1"; shift; ;;
        -q|--qp) shift; QP="$1"; shift; ;;
        -s|--seq_name) shift; SEQ="$1"; shift; ;;
        -x|--extra_params) shift; PIPELINE_PARAMS="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

export DNNL_MAX_CPU_ISA=AVX2
export DEVICE=${DEVICE}

DATASET_SRC="${FCM_TESTDATA}/mpeg-oiv6"

CONF_NAME="eval_split_inference_example.yaml"
if [[ ${PIPELINE} == "remote" ]]; then
  CONF_NAME="eval_remote_inference_example.yaml"
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
echo "Pipeline Type:      " ${PIPELINE} " Video"
echo "Datatset location:  " ${FCM_TESTDATA}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "hm"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Input sequence:     " ${SEQ}
echo "Seq. Framerate:     " ${FRAME_RATE}
echo "QP for Inner Codec: " ${QP}
echo "Intra Period for Inner Codec: "${INTRA_PERIOD}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="
 
compressai-${PIPELINE}-inference --config-name=${CONF_NAME} \
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
        ${PIPELINE_PARAMS} \
