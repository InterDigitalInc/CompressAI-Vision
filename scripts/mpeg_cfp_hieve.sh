#!/usr/bin/env bash
set -eu

EXPERIMENT="_hieve_cfp_test"
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


QP_ANCHORS_HIEVE_13=`echo "20 22 24 26 28 29"`
QP_ANCHORS_HIEVE_16=`echo "22 24 26 28 30 31"`
QP_ANCHORS_HIEVE_01=`echo "22 25 27 29 31 34"`
QP_ANCHORS_HIEVE_17=`echo "22 23 24 26 27 28"`
QP_ANCHORS_HIEVE_18=`echo "22 25 27 29 31 34"`


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
HIEVE_SRC="${VCM_TESTDATA}/HiEve_pngs"

# HIEVE - Object Tracking with JDE
for SEQ in \
            '13' \
            '16' \
            '2' \
            '17' \
            '18'
do
    for qp in ${QPS}
    do
        if [ ${USE_GRID} == "True" ]; then
            JOBNAME="${CONF_NAME}-mpeg-hieve-tracking-qp${qp}"
            CMD="${CMD_GRID} --job-name=${JOBNAME} -- compressai-vision-eval"
        fi
        ${CMD} --config-name=${CONF_NAME}.yaml ${CODEC_PARAMS} \
                    ++pipeline.type=video \
                    ++paths._runs_root=${OUTPUT_DIR} \
                    ++pipeline.conformance.save_conformance_files=True \
                    ++pipeline.conformance.subsample_ratio=90 \
                    ++codec.encoder_config.feature_channel_suppression.manual_cluster=True \
                    ++codec.encoder_config.feature_channel_suppression.n_clusters='{105: 20, 90: 20, 75: 20}' \
                    ++codec.encoder_config.feature_channel_suppression.downscale=False \
                    ++codec.encoder_config.qp=${qp} \
                    ++codec.eval_encode='bitrate' \
                    ++codec.experiment=${EXPERIMENT} \
                    ++vision_model.arch=jde_1088x608 \
                    ++vision_model.jde_1088x608.splits="[105, 90, 75]" \
                    ++dataset.type=TrackingDataset \
                    ++dataset.settings.patch_size="[608, 1088]" \
                    ++dataset.datacatalog=MPEGHIEVE \
                    ++dataset.config.root=${HIEVE_SRC}/${SEQ} \
                    ++dataset.config.imgs_folder=img1 \
                    ++dataset.config.annotation_file=gt/gt.txt \
                    ++dataset.config.dataset_name=mpeg-hieve-${SEQ} \
                    ++evaluator.type=MOT-HIEVE-EVAL \
                    ++misc.device="${DEVICE}"
    done
done

wait
python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset HIEVE --result_path ${OUTPUT_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/MPEGHIEVE/