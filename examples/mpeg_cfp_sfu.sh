#!/usr/bin/env bash

set -eu

OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENTRY_CMD="compressai-vision-eval"

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
    for qp in {5..10}
    do
        ${ENTRY_CMD} --config-name=eval_cfp_codec.yaml \
                    ++pipeline.type=video \
                    ++paths._runs_root=${OUTPUT_DIR} \
                    ++pipeline.conformance.save_conformance_files=True \
                    ++pipeline.conformance.subsample_ratio=9 \
                    ++codec.encoder_config.qp=${qp} \
                    ++codec.eval_encode='bitrate' \
                    ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
                    ++dataset.type=Detectron2Dataset \
                    ++dataset.datacatalog=SFUHW \
                    ++dataset.config.root=${SFU_HW_SRC}/${SEQ} \
                    ++dataset.config.annotation_file=annotations/${SEQ}.json \
                    ++dataset.config.dataset_name=sfu-hw-${SEQ} \
                    ++evaluator.type=COCO-EVAL
    done
done

python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset SFU --result_path ${OUTPUT_DIR}/split-inference-video/cfp_codec/SFUHW/