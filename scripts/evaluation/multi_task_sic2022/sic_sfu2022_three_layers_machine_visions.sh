#!/usr/bin/env bash
set -eu

DATASET_DIR=""
OUTPUT_DIR=""
EXPERIMENT="test"
DEVICE="cuda"
QID=1
TLID=0
PIPELINE_PARAMS=""


#parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_
Runs an evaluation pipeline (split or remote) using VTM as codec

RUN OPTIONS:
                [-t|--testdata) path to the test dataset, e.g. path/to/coco2017/, default=""]
                [-o|--output_dir) root of output folder for logs and artifacts, default=""]
                [-e|--exp_name) name of the experiments to locate/label outputs, default="test"]
                [-d|--device) cuda or cpu, default="cpu"]
                [-q|--qid) quality level, depends on the learned codec, default=1]
                [-l|--tlid) target layer id, default=0]
                [-x|--extra_params) additional parameters to override default configs (pipeline/codec/evaluation...), default=""]
EXAMPLE         [bash eval_on_hieve_vtm.sh -t /path/to/testdata -p split -i /path/to/VTM_repo -d cpu -q 32 -s Traffic_2560x1600_30_val]
_EOF_
            exit;
            ;;
        -t|--testdata) shift; DATASET_DIR="$1"; shift; ;;
        -o|--output_dir) shift; OUTPUT_DIR="$1"; shift; ;;
        -e|--exp_name) shift; EXPERIMENT="$1"; shift; ;;
        -d|--device) shift; DEVICE="$1"; shift; ;;
        -q|--qid) shift; QID="$1"; shift; ;;
        -l|--tlid) shift; TLID="$1"; shift; ;;
        -x|--extra_params) shift; PIPELINE_PARAMS="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

export DNNL_MAX_CPU_ISA=AVX2
export DEVICE=${DEVICE}


echo "============================== RUNNING COMPRESSAI-VISION EVAL== =================================="
echo "Pipeline Type:      " "Multi-task inference "
echo "Datatset location:  " ${DATASET_DIR}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "multi-task"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Target Layer ID:    " ${TLID}
echo "Quality Index:      " ${QID}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="



CONF_NAME="eval_multitask_inference_example"

CMD="compressai-multi-task-inference"

${CMD} --config-name=${CONF_NAME}.yaml \
        ++paths._run_root=${OUTPUT_DIR} \
        ++pipeline.codec.num_tasks=3 \
        ++pipeline.codec.target_task_layer=${TLID} \
        ++codec.bitstream_name=sic_sfu2023_${QID}_tl_${TLID} \
        ++codec.encoder_config.qidx=${QID} \
        ++codec.experiment=${EXPERIMENT} \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=COCO \
        ++dataset.config.root=${DATASET_DIR} \
        ++dataset.config.dataset_name=coco2017val \
        ++dataset.config.imgs_folder=val2017 \
        ++dataset.config.annotation_file=annotations/instances_val2017.json \
        ++dataset.config.seqinfo=None \
        ++misc.device=${DEVICE} \
        ${PIPELINE_PARAMS} \


#++dataset.type=DefaultDataset
#++dataset.datacatalog=IMAGES
#++dataset.config.root=/mnt/wekamount/RI-Projects/library_ds_datasets/kodak/
#++dataset.config.dataset_name=kodak
#++dataset.config.imgs_folder=test
#++dataset.config.annotation_file=None
#++pipeline.codec.encode_only=False
#++pipeline.codec.decode_only=True
