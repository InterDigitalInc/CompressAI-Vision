#!/usr/bin/env bash
#
# This runs the evaluation of original models, whitout compression 
# make sure you sourced the virtual environment that contains up-to-date installed compressai-vision
# see provided installation scripts
set -eu

usage() {
    echo "Usage: $0 --command <command> --testdata <path> --device <device>"
    echo ""
    echo "Runs evaluation for mmpose rtmo."
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

export TORCH_HOME=/tmp/${USER}/torch-cache
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
export REQUESTS_CA_BUNDLE=${SSL_CERT_FILE}

# /path/to/COCODataset/
COCO_2017_VAL_SRC="${TESTDATA_DIR}/coco2017"

# COCO 2017 Val - Segmentation
# Bottom-up method

# option for split points "backbone" only
${ENTRY_CMD} --config-name=${CONF_NAME}.yaml \
             pipeline.type=image \
             paths._run_root=/tmp/cav-lraspp_mobilenet_v3-coco-val2017 \
             vision_model.arch=lraspp_mobilenet_v3_large \
             dataset.type=Detectron2Dataset \
             dataset.settings.linear_mapper=True \
             dataset.settings.preserve_annotations=True \
             dataset.datacatalog=COCO \
             dataset.config.root=${COCO_2017_VAL_SRC} \
             dataset.config.imgs_folder=val2017 \
             dataset.config.annotation_file=annotations/instances_val2017.json \
             dataset.config.dataset_name=coco-val2017-voc-labels \
             dataset.loader.num_workers=0 \
             evaluator.type=TORCHVISION-SEMSEG-EVAL \
             evaluator.eval_criteria=mIoU \
             evaluator.num_classes=21 \
             evaluator.ignore_index=255 \
             evaluator.category_id_mapping=coco_to_voc \
             evaluator.overwrite_results=True \
             codec.eval_encode=bpp \
             misc.device.nn_parts=${DEVICE}
  
#split-inference only

             #pipeline.conformance.save_conformance_files=False \
             #pipeline.nn_task_part1.load_features=False \
             #pipeline.nn_task_part1.dump_features=False \
             #pipeline.nn_task_part2.dump_features=False \