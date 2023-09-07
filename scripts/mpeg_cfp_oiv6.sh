#!/usr/bin/env bash
set -eu

EXPERIMENT="_oiv6_cfp_test" # e.g. "_clipping_on"

### tune the following parameters
# OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/racapef/runs"
OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs"

USE_GRID="True"
DEVICE="cuda"

CONF_NAME="eval_cfp_codec"
# CONF_NAME="eval_vtm"
# CONF_NAME="eval_ffmpeg"

CODEC_PARAMS=""
# e.g.
# CODEC_PARAMS="++codec.type=x265"

QP_ANCHORS_OIV6=`echo "35 37 39 41 43 45"`
# QPS=${QP_ANCHORS_OIV6}
QPS=`echo "7 8 9 10 11 12"`
###

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CMD="compressai-vision-eval"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs

if [ ${DEVICE} == "cuda" ]; then
    CMD_GRID="grid batch --reservation=deepvideo \
              --gpus 1 --cpus 1 -e ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.err -o ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.out"
else
    CMD_GRID="grid batch --nodelist=kopspgd16p \
              --cpus 1 -e ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.err -o ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.out"
fi

VCM_TESTDATA="/data/datasets/MPEG-FCVCM/vcm_testdata"
MPEG_OIV6_SRC="${VCM_TESTDATA}/mpeg-oiv6"

for qp in ${QPS}
do
    if [ ${USE_GRID} == "True" ]; then
        JOBNAME="${CONF_NAME}-mpeg-oiv6-objdet-qp${qp}"
        CMD="${CMD_GRID} --job-name=${JOBNAME} -- compressai-vision-eval"
    fi
    ${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
            ++paths._runs_root=${OUTPUT_DIR} \
            ++pipeline.type=image \
            ++pipeline.conformance.save_conformance_files=True \
            ++pipeline.conformance.subsample_ratio=9 \
            ++codec.encoder_config.n_cluster='{"p2": 128, "p3": 128, "p4": 150, "p5": 180}' \
            ++codec.encoder_config.qp=${qp} \
            ++codec.eval_encode='bpp' \
            ++codec.experiment=${EXPERIMENT} \
            ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
            ++dataset.type=Detectron2Dataset \
            ++dataset.datacatalog=MPEGOIV6 \
            ++dataset.config.root=${MPEG_OIV6_SRC} \
            ++dataset.config.annotation_file=annotations/mpeg-oiv6-detection-coco.json \
            ++dataset.config.dataset_name=mpeg-oiv6-detection \
            ++evaluator.type=OIC-EVAL \
            ++misc.device="${DEVICE}"


    if [ ${USE_GRID} == "True" ]; then
        JOBNAME="${CONF_NAME}-mpeg-oiv6-objseg-qp${qp}"
        CMD="${CMD_GRID} --job-name=${JOBNAME} -- compressai-vision-eval"
    fi
    echo "running ${JOBNAME} with qp=${qp}"  
    ${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
        ++paths._runs_root="${OUTPUT_DIR}" \
        ++pipeline.type=image \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
        ++codec.encoder_config.n_cluster='{"p2": 128, "p3": 128, "p4": 150, "p5": 180}' \
        ++codec.encoder_config.qp=${qp} \
        ++codec.eval_encode='bpp' \
        ++codec.experiment=${EXPERIMENT} \
        ++vision_model.arch=mask_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=MPEGOIV6 \
        ++dataset.config.root=${MPEG_OIV6_SRC} \
        ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco.json \
        ++dataset.config.dataset_name=mpeg-oiv6-segmentation \
        ++evaluator.type=OIC-EVAL \
        ++misc.device="${DEVICE}"
done

wait 
python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset OIV6 --result_path ${OUTPUT_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/
