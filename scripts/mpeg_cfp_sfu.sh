#!/usr/bin/env bash
set -eu

### tune the following parameters
OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/racapef/runs"
# OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs"

USE_GRID="True"
DEVICE="cuda"

CODEC="cfp_codec"
CODEC_PARAMS=""
# e.g.
# CODEC_PARAMS="++codec.encoder_config.clipping=True"

EXPERIMENT="_test3" # e.g. "_clipping_on"

QPS=`echo "5 6 7 8 9 10"`
###

CONF_NAME="eval_${CODEC}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${OUTPUT_DIR}/slurm_logs

CMD="compressai-vision-eval"
if [ ${DEVICE} == "cuda" ]; then
    CMD_GRID="grid batch --reservation=deepvideo \
              --gpus 1 --cpus 1 -e ${OUTPUT_DIR}/slurm_logs/slurm-%j.err -o ${OUTPUT_DIR}/slurm_logs/slurm-%j.out"
else
    CMD_GRID="grid batch --nodelist=kopspgd16p \
              --cpus 1 -e ${OUTPUT_DIR}/slurm_logs/slurm-%j.err -o ${OUTPUT_DIR}/slurm_logs/slurm-%j.out"
fi

VCM_TESTDATA="/data/datasets/MPEG-FCVCM/vcm_testdata"
SFU_HW_SRC="${VCM_TESTDATA}/SFU_HW_Obj"

for SEQ in \
            'Traffic_2560x1600_30_val' \
            'Kimono_1920x1080_24_val' \
            'ParkScene_1920x1080_24_val' \
            'Cactus_1920x1080_50_val' \
            'BasketballDrive_1920x1080_50_val' \
            'BQTerrace_1920x1080_60_val' \
            'BasketballDrill_832x480_50_val' \
            'BQMall_832x480_60_val' \
            'PartyScene_832x480_50_val' \
            'RaceHorses_832x480_30_val' \
            'BasketballPass_416x240_50_val' \
            'BQSquare_416x240_60_val' \
            'BlowingBubbles_416x240_50_val' \
            'RaceHorses_416x240_30_val'
do
    for qp in ${QPS}
    do
        if [ ${USE_GRID} == "True" ]; then
            JOBNAME="${CONF_NAME}-mpeg-oiv6-objdet-qp${qp}"
            CMD="${CMD_GRID} --job-name=${JOBNAME} -- compressai-vision-eval"
        fi
        ${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
                    ++pipeline.type=video \
                    ++paths._runs_root=${OUTPUT_DIR} \
                    ++pipeline.conformance.save_conformance_files=True \
                    ++pipeline.conformance.subsample_ratio=9 \
                    ++codec.encoder_config.qp=${qp} \
                    ++codec.eval_encode='bitrate' \
                    ++codec.experiment=${EXPERIMENT} \
                    ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
                    ++dataset.type=Detectron2Dataset \
                    ++dataset.datacatalog=SFUHW \
                    ++dataset.config.root=${SFU_HW_SRC}/${SEQ} \
                    ++dataset.config.annotation_file=annotations/${SEQ}.json \
                    ++dataset.config.dataset_name=sfu-hw-${SEQ} \
                    ++evaluator.type=COCO-EVAL \
                    ++misc.device="${DEVICE}"
    done
done

wait
python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset SFU --result_path ${OUTPUT_DIR}/split-inference-video/${CODEC}${EXPERIMENT}/SFUHW/