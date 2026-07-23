#!/usr/bin/env bash
#
# This runs the evaluation of original models, whitout compression
# make sure you sourced the virtual environment that contains up-to-date installed compressai-vision
# see provided installation scripts
set -eu

usage() {
    echo "Usage: $0 --command <command> --testdata <path> --device <device>"
    echo ""
    echo "Runs evaluation for yolox darknet3 performance."
    echo ""
    echo "Options:"
    echo "  -c, --command      Entrypoint command. Options: compressai-split-inference, compressai-remote-inference"
    echo "  -t, --testdata     Path to the test data directory (e.g., /path/to/COCODataset/)."
    echo "  -d, --device       Device to use for evaluation (e.g., cuda:0)."
    echo "  -h, --help         Display this help message."
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--command) shift; ENTRY_CMD="$1"; shift; ;;
        -o|--output_dir) shift; OUTPUT_DIR="$1"; shift; ;;
        -t|--testdata) shift; TESTDATA_DIR="$1"; shift; ;;
        -s|--seq_name) shift; SEQ="$1"; shift; ;;
        -d|--device) shift; DEVICE="$1"; shift; ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown parameter $1"; usage; exit; ;;
    esac
done

# Check if mandatory arguments are provided
if [ -z "${ENTRY_CMD-}" ] || [ -z "${TESTDATA_DIR-}" ] || [ -z "${DEVICE-}" ]; then
    echo "Error: Missing mandatory arguments."
    usage
fi

# List of entry cmds
CMD_OPTS=("compressai-split-inference" "compressai-remote-inference")

if [[ " ${CMD_OPTS[@]} " =~ " ${ENTRY_CMD} " ]]; then
    echo "Run ${ENTRY_CMD} ........"
else
    echo ": ${ENTRY_CMD} does not exist in the options."
    echo ": Please choose one out of these options: ${CMD_OPTS[*]}"
    exit 1
fi

declare -A configs

configs["compressai-split-inference"]="eval_split_inference_example"
configs["compressai-remote-inference"]="eval_remote_inference_example"

CONF_NAME=${configs[${ENTRY_CMD}]}

if [ ! -d "${TESTDATA_DIR}" ]; then
    echo "${TESTDATA_DIR} does not exist, please select dataset folder, e.g.
    $ bash default_vision_performances.sh --command [entry_cmd] --testdata [/path/to/dataset] --device [device]"
    exit
fi

export DNNL_MAX_CPU_ISA=AVX2
export DEVICE=${DEVICE}


DATASET_SRC="${TESTDATA_DIR}/SFU_HW_Obj"

PREFIX=""
AUGMENT_BYPASS=False

declare -A non_rescale_dict
non_rescale_dict["ns_Traffic_2560x1600_30_val"]="Traffic_2560x1600_30_val"
non_rescale_dict["ns_BQTerrace_1920x1080_60_val"]="BQTerrace_1920x1080_60_val"

if [[ ${non_rescale_dict[$SEQ]+_} ]]; then
    echo "Remapped sequence: " ${SEQ}
    SEQ=${non_rescale_dict[$SEQ]}
    PREFIX="ns_"
    AUGMENT_BYPASS=True
fi

# option for split points "l13" or "l37"
# ++vision_model.yolox_darknet53.splits="l37" \
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             pipeline.type=video \
             paths._run_root=${OUTPUT_DIR} \
             pipeline.conformance.save_conformance_files=False \
             vision_model.arch=yolox_darknet53 \
             dataset.type=YOLOXDataset \
             dataset.settings.patch_size="[640, 640]" \
             dataset.datacatalog=SFUHW \
             dataset.config.root=${DATASET_SRC}/${SEQ} \
             dataset.config.annotation_file=annotations/${SEQ}.json \
             dataset.config.dataset_name=sfu-hw-${PREFIX}${SEQ} \
             dataset.settings.input_augmentation_bypass=${AUGMENT_BYPASS} \
             codec.encoder_config.qp=42 \
             evaluator.type=YOLOX-COCO-EVAL \
             evaluator.overwrite_results=True \
             pipeline.nn_task_part1.load_features=False \
             pipeline.nn_task_part1.dump_features=False \
             pipeline.nn_task_part2.dump_features=False \
             codec.eval_encode=bitrate \
             misc.device.nn_parts=${DEVICE}

