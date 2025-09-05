#!/usr/bin/env bash
#
# This runs the evaluation of original models, whitout compression 
# make sure you sourced the virtual environment that contains up-to-date installed compressai-vision
# see provided installation scripts
set -eu

usage() {
    echo "Usage: $0 --command <command> --testdata <path> --device <device>"
    echo ""
    echo "Runs evaluation for vision performances with SAM."
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
        -t|--testdata) shift; TESTDATA_DIR="$1"; shift; ;;
        -d|--device) shift; DEVICE="$1"; shift; ;;
        -h|--help) usage;;
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

MPEG_SAM="${TESTDATA_DIR}/sam_test"



# MPEGOIV6 - Small test
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             ++pipeline.type=image \
             ++vision_model.arch=sam_vit_h_4b8939 \
             ++dataset.type=SamDataset \
             ++dataset.datacatalog=MPEGSAM \
             ++dataset.config.root=${MPEG_SAM} \
             ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco_2images.json \
             ++dataset.config.dataset_name=mpeg-oiv6-sam \
             ++evaluator.type=OIC-EVAL \
             ++evaluator.overwrite_results=True \
             ++codec.encoder_config.parallel_encoding=False \
             ++pipeline.nn_task_part1.load_features=False \
             ++pipeline.nn_task_part1.dump_features=False \
             ++pipeline.nn_task_part2.dump_features=False \
             ++misc.device.nn_parts=${DEVICE} \
             ++codec.eval_encode='bpp' \

             # ++pipeline.conformance.save_conformance_files=True \
             # ++pipeline.conformance.subsample_ratio=9 \


