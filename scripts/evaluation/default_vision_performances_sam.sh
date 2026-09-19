#!/usr/bin/env bash
#
# Runs SAM inference on the MPEG-OIV6 image dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/models/segment_anything${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    cat <<EOF
Usage: $0 --command <command> --testdata <dataset-dir> --device <device> [options]

Options:
  -c, --command      compressai-split-inference or compressai-remote-inference.
  -t, --testdata     MPEG-OIV6 directory containing images/, prompts/, and annotations/.
  -d, --device       Device used for SAM inference, for example cuda:0 or cpu.
  -o, --output-dir   Root output directory (default: ./logs/runs).
  -s, --split-point  SAM split point: imgenc or global_attn2 (default: imgenc).
  -a, --annotation   Annotation path relative to the dataset root
                     (default: annotations/mpeg-oiv6-segmentation-coco.json).
      --dry-run      Resolve and print the Hydra configuration without running inference.
      --             Pass all remaining arguments as Hydra overrides.
  -h, --help         Display this help message.

Example:
  $0 \
    --command compressai-split-inference \
    --testdata /data/datasets/MPEG-FCM/fcm_testdata/mpeg-oiv6 \
    --device cuda:0
EOF
}

OUTPUT_DIR="./logs/runs"
SPLIT_POINT="imgenc"
ANNOTATION_FILE="annotations/mpeg-oiv6-segmentation-coco.json"
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--command) ENTRY_CMD="$2"; shift 2 ;;
        -t|--testdata) TESTDATA_DIR="$2"; shift 2 ;;
        -d|--device) DEVICE="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -s|--split-point) SPLIT_POINT="$2"; shift 2 ;;
        -a|--annotation) ANNOTATION_FILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown parameter: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${ENTRY_CMD:-}" || -z "${TESTDATA_DIR:-}" || -z "${DEVICE:-}" ]]; then
    echo "[ERROR] --command, --testdata, and --device are required." >&2
    usage
    exit 1
fi

case "${ENTRY_CMD}" in
    compressai-split-inference) CONF_NAME="eval_split_inference_example.yaml" ;;
    compressai-remote-inference) CONF_NAME="eval_remote_inference_example.yaml" ;;
    *) echo "[ERROR] Unsupported command: ${ENTRY_CMD}" >&2; exit 1 ;;
esac

if ! command -v "${ENTRY_CMD}" >/dev/null 2>&1; then
    echo "[ERROR] Command not found: ${ENTRY_CMD}" >&2
    echo "Activate the CompressAI-Vision environment before running this script." >&2
    exit 1
fi

for directory in images prompts; do
    if [[ ! -d "${TESTDATA_DIR}/${directory}" ]]; then
        echo "[ERROR] Directory does not exist: ${TESTDATA_DIR}/${directory}" >&2
        exit 1
    fi
done

if [[ ! -f "${TESTDATA_DIR}/${ANNOTATION_FILE}" ]]; then
    echo "[ERROR] Annotation file does not exist: ${TESTDATA_DIR}/${ANNOTATION_FILE}" >&2
    exit 1
fi

if [[ "${SPLIT_POINT}" != "imgenc" && "${SPLIT_POINT}" != "global_attn2" ]]; then
    echo "[ERROR] Unsupported split point: ${SPLIT_POINT}" >&2
    exit 1
fi

declare -a COMMAND=(
    "${ENTRY_CMD}"
    "--config-name=${CONF_NAME}"
    "pipeline.type=image"
    "paths._run_root=${OUTPUT_DIR}"
    "vision_model.arch=sam_vit_h_4b8939"
    "vision_model.sam_vit_h_4b8939.splits=${SPLIT_POINT}"
    "dataset.type=SamDataset"
    "dataset.datacatalog=MPEGSAM"
    "dataset.config.root=${TESTDATA_DIR}"
    "dataset.config.imgs_folder=images"
    "dataset.config.prompts_folder=${TESTDATA_DIR}/prompts"
    "dataset.config.prompt_format=point_label_class"
    "dataset.config.annotation_file=${ANNOTATION_FILE}"
    "dataset.config.dataset_name=mpeg-oiv6-sam"
    "dataset.config.seqinfo=none"
    "dataset.config.ext=jpg"
    "dataset.loader.num_workers=1"
    "evaluator.type=OIC-EVAL"
    "evaluator.overwrite_results=True"
    "codec.type=bypass"
    "codec.eval_encode=bpp"
    "codec.device=${DEVICE}"
    "+codec.hash_dir=${OUTPUT_DIR}/hashes"
    "+codec.save_visualization=False"
    "pipeline.nn_task_part1.load_features=False"
    "pipeline.nn_task_part1.dump_features=False"
    "pipeline.nn_task_part2.dump_features=False"
    "misc.device.nn_parts=${DEVICE}"
)

COMMAND+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" == true ]]; then
    COMMAND+=("--cfg" "job" "--resolve")
fi

echo "Dataset:     ${TESTDATA_DIR}"
echo "Split point: ${SPLIT_POINT}"
echo "Device:      ${DEVICE}"
echo "Prompts:     ${TESTDATA_DIR}/prompts"
echo "Annotation:  ${TESTDATA_DIR}/${ANNOTATION_FILE}"

"${COMMAND[@]}"
