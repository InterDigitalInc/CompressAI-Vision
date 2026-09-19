#!/usr/bin/env bash
#
# Runs SAM split inference on an image sequence using the video pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/models/segment_anything${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    cat <<EOF
Usage: $0 --testdata <sequence-dir> --device <device> [options]

Options:
  -t, --testdata     Sequence directory containing images/, prompts/, and annotations/.
  -d, --device       Device used for SAM inference, for example cuda:0 or cpu.
  -o, --output-dir   Root output directory (default: ./logs/runs).
  -s, --split-point  SAM split point: imgenc or global_attn2 (default: global_attn2).
      --command      Entrypoint command (default: compressai-split-inference).
      --dry-run      Resolve and print the Hydra configuration without running inference.
      --             Pass all remaining arguments as Hydra overrides.
  -h, --help         Display this help message.

Example:
  $0 \
    --testdata /data/datasets/MPEG-FCM/fcm_testdata/SFU_HW_Obj_sam_5seq/Traffic_2560x1600_30_val \
    --device cuda:0
EOF
}

ENTRY_CMD="compressai-split-inference"
OUTPUT_DIR="./logs/runs"
SPLIT_POINT="global_attn2"
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--testdata) TESTDATA_DIR="$2"; shift 2 ;;
        -d|--device) DEVICE="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -s|--split-point) SPLIT_POINT="$2"; shift 2 ;;
        --command) ENTRY_CMD="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown parameter: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${TESTDATA_DIR:-}" || -z "${DEVICE:-}" ]]; then
    echo "[ERROR] --testdata and --device are required." >&2
    usage
    exit 1
fi

if [[ ! -d "${TESTDATA_DIR}/images" ]]; then
    echo "[ERROR] Image directory does not exist: ${TESTDATA_DIR}/images" >&2
    exit 1
fi

if [[ ! -d "${TESTDATA_DIR}/prompts" ]]; then
    echo "[ERROR] Prompt directory does not exist: ${TESTDATA_DIR}/prompts" >&2
    exit 1
fi

if [[ "${SPLIT_POINT}" != "imgenc" && "${SPLIT_POINT}" != "global_attn2" ]]; then
    echo "[ERROR] Unsupported split point: ${SPLIT_POINT}" >&2
    exit 1
fi

if ! command -v "${ENTRY_CMD}" >/dev/null 2>&1; then
    echo "[ERROR] Command not found: ${ENTRY_CMD}" >&2
    echo "Activate the CompressAI-Vision environment before running this script." >&2
    exit 1
fi

SEQ_NAME="$(basename "${TESTDATA_DIR}")"
ANNOTATION_FILE="annotations/${SEQ_NAME}_seg_fixed.json"

if [[ ! -f "${TESTDATA_DIR}/${ANNOTATION_FILE}" ]]; then
    echo "[ERROR] Annotation file does not exist: ${TESTDATA_DIR}/${ANNOTATION_FILE}" >&2
    exit 1
fi

declare -a COMMAND=(
    "${ENTRY_CMD}"
    "--config-name=eval_split_inference_example.yaml"
    "pipeline.type=video"
    "paths._run_root=${OUTPUT_DIR}"
    "vision_model.arch=sam_vit_h_4b8939"
    "vision_model.sam_vit_h_4b8939.splits=${SPLIT_POINT}"
    "dataset.type=SamDataset"
    "dataset.datacatalog=MPEGSAM"
    "dataset.config.root=${TESTDATA_DIR}"
    "dataset.config.imgs_folder=images"
    "dataset.config.prompts_folder=${TESTDATA_DIR}/prompts"
    "dataset.config.prompt_format=points_class"
    "dataset.config.annotation_file=${ANNOTATION_FILE}"
    "dataset.config.dataset_name=sfu-hw-${SEQ_NAME}-sam"
    "dataset.config.seqinfo=seqinfo.ini"
    "dataset.config.ext=png"
    "dataset.loader.num_workers=1"
    "evaluator.type=COCO-EVAL"
    "+evaluator.tasks=[bbox]"
    "evaluator.eval_criteria=AP"
    "evaluator.overwrite_results=True"
    "codec.type=bypass"
    "codec.eval_encode=bitrate"
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

echo "Sequence:    ${SEQ_NAME}"
echo "Split point: ${SPLIT_POINT}"
echo "Device:      ${DEVICE}"
echo "Prompts:     ${TESTDATA_DIR}/prompts"
echo "Annotation:  ${TESTDATA_DIR}/${ANNOTATION_FILE}"

"${COMMAND[@]}"
