#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
Runs an evaluation pipeline (split or remote) using VTM as codec

RUN OPTIONS:
                [-t|--testdata) path to the test dataset, e.g. path/to/fcm_testdata/, default=""]
                [-p|--pipeline) type of pipeline, split/remote, default="split"]
                [-i|--inner_codec) path to core codec, e.g. /path/to/VTM_repo (that contains subfolders bin/ cfg/ ...), default=""]
                [-o|--output_dir) root of output folder for logs and artifacts, default=""]
                [-e|--exp_name) name of the experiments to locate/label outputs, default="test"]
                [-d|--device) cuda or cpu, default="cpu"]
                [-q|--qp) quality level, depends on the inner codec, default=42]
                [-s|--seq_name) sequence name as used in testdata root folder. E.g., "Traffic_2560x1600_30_val" in sfu_hw_obj, default="42"]
                [-x|--extra_params) additional parameters to override default configs (pipeline/codec/evaluation...), default=""]
EXAMPLE         [bash eval_on_mpeg_sfu_hw_vcmrs.sh -t /path/to/testdata -p split -i /path/to/VTM_repo -d cpu -q 32 -s Traffic_2560x1600_30_val]
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

DATASET_SRC="${FCM_TESTDATA}/SFU_HW_Obj"

CONF_NAME="eval_split_inference_example.yaml"
if [[ ${PIPELINE} == "remote" ]]; then
  CONF_NAME="eval_remote_inference_example.yaml"
fi

DATASET_INFO_PATH="${SCRIPT_DIR}/sfu_hw_obj.json"
SEQ_INFO=$(jq --compact-output ".sequences[] | select(.seq_name == \"${SEQ}\")" "$DATASET_INFO_PATH")
[ -n "$SEQ_INFO" ] || exit 1
INTRA_PERIOD=$(jq --raw-output ".intra_period" <<< "${SEQ_INFO}")
FRAME_RATE=$(jq --raw-output ".frame_rate" <<< "${SEQ_INFO}")

echo "============================== RUNNING COMPRESSAI-VISION EVAL== =================================="
echo "Pipeline Type:      " ${PIPELINE} " Video"
echo "Datatset location:  " ${FCM_TESTDATA}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "vcmrs"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Input sequence:     " ${SEQ}
echo "Seq. Framerate:     " ${FRAME_RATE}
echo "QP for Inner Codec: " ${QP}
echo "Intra Period for Inner Codec: "${INTRA_PERIOD}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="

compressai-${PIPELINE}-inference --config-name=${CONF_NAME} \
        ++pipeline.type=video \
	++pipeline.codec.vcm_mode=True \
	++pipeline.codec.output10b=True \
        ++paths._run_root=${OUTPUT_DIR} \
	++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=SFUHW \
        ++dataset.config.root=${DATASET_SRC}/${SEQ} \
        ++dataset.config.annotation_file=annotations/${SEQ}.json \
        ++dataset.config.dataset_name=sfu-hw-${SEQ} \
        ++evaluator.type=COCO-EVAL \
        ++codec.experiment=${EXPERIMENT} \
	codec=vcmrs.yaml \
        ++codec.encoder_config.intra_period=${INTRA_PERIOD} \
        ++codec.encoder_config.parallel_encoding=True \
        ++codec.encoder_config.qp=${QP} \
        ++codec.encoder_config.input_bitdepth=8 \
        ++codec.eval_encode='bitrate' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device.nn_parts=${DEVICE} \
        ${PIPELINE_PARAMS} \
