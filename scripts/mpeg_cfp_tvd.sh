#!/usr/bin/env bash
set -eu

EXPERIMENT="_tvd_cfp_test"
### tune the following parameters
OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs"
# OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs"

USE_GRID="True"
DEVICE="cuda"

CONF_NAME="eval_cfp_codec"
# CONF_NAME="eval_vtm"
# CONF_NAME="eval_ffmpeg"

CODEC_PARAMS=""
# e.g.
# CODEC_PARAMS="++codec.type=x265"


QP_ANCHORS_TVD_01=`echo "22 24 26 29 31 33"`
QP_ANCHORS_TVD_02=`echo "23 25 27 28 30 31"`
QP_ANCHORS_TVD_03=`echo "25 26 27 29 30 31"`


QPS=`echo "5 6 7 8 9"`
###

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs

CMD="compressai-vision-eval"
if [ ${DEVICE} == "cuda" ]; then
    CMD_GRID="grid batch --reservation=deepvideo \
              --gpus 1 --cpus 1 -e ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.err -o ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.out"
else
    CMD_GRID="grid batch --nodelist=kopspgd16p \
              --cpus 1 -e ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.err -o ${OUTPUT_DIR}/${EXPERIMENT}/slurm_logs/slurm-%j.out"
fi

VCM_TESTDATA="/data/datasets/MPEG-FCVCM/vcm_testdata"
TVD_SRC="${VCM_TESTDATA}/tvd_tracking"

# TVD - Object Tracking with JDE
for SEQ in \
            'TVD-01' \
            'TVD-02' \
            'TVD-03'
do
    for qp in ${QPS}
    do
        if [ ${USE_GRID} == "True" ]; then
            JOBNAME="${CONF_NAME}-mpeg-tvd-tracking-qp${qp}"
            CMD="${CMD_GRID} --job-name=${JOBNAME} -- compressai-vision-eval"
        fi
        ${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
                    ++pipeline.type=video \
                    ++paths._runs_root=${OUTPUT_DIR} \
                    ++pipeline.conformance.save_conformance_files=True \
                    ++pipeline.conformance.subsample_ratio=90 \
                    ++codec.encoder_config.n_cluster='{74: 20, 61: 20, 36: 20}' \
                    ++codec.encoder_config.qp=${qp} \
                    ++codec.eval_encode='bitrate' \
                    ++codec.experiment=${EXPERIMENT} \
                    ++vision_model.arch=jde_1088x608 \
                    ++vision_model.jde_1088x608.splits="[74, 61, 36]" \
                    ++dataset.type=TrackingDataset \
                    ++dataset.settings.patch_size="[608, 1088]" \
                    ++dataset.datacatalog=MPEGTVDTRACKING \
                    ++dataset.config.root=${TVD_SRC}/${SEQ} \
                    ++dataset.config.imgs_folder=img1 \
                    ++dataset.config.annotation_file=gt/gt.txt \
                    ++dataset.config.dataset_name=mpeg-tracking-${SEQ} \
                    ++evaluator.type=MOT-TVD-EVAL \
                    ++misc.device="${DEVICE}"
    done
done

wait
python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset TVD --result_path ${OUTPUT_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGTVDTRACKING/