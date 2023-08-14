#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENTRY_CMD="${SCRIPT_DIR}/../compressai_vision/run/eval_split_inference.py"

VCM_TESTDATA="${SCRIPT_DIR}/../../vcm_testdata"

if [ $# == 1 ]; then
    VCM_TESTDATA=$1
fi
if [ ! -d "${VCM_TESTDATA}" ]; then
    echo "${VCM_TESTDATA} does not exist, please select dataset folder, e.g.
    $ bash default_vision_performances.sh  /data/vcm_dataset"
    exit
fi

MPEG_OIV6_SRC="${VCM_TESTDATA}/mpeg-oiv6"
SFU_HW_SRC="${VCM_TESTDATA}/SFU_HW_Obj"
HIEVE_SRC="${VCM_TESTDATA}/HiEve_pngs"
TVD_SRC="${VCM_TESTDATA}/tvd_tracking"

# MPEGOIV6 - Detection with Faster RCNN
python ${ENTRY_CMD} --config-name=eval_example.yaml \
                    ++pipeline.type=unfold \
                    ++pipeline.conformance.save_conformance_files=True \
                    ++pipeline.conformance.subsample_ratio=9 \
                    ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
                    ++dataset.type=Detectron2Dataset \
                    ++dataset.datacatalog=MPEGOIV6 \
                    ++dataset.config.root=${MPEG_OIV6_SRC} \
                    ++dataset.config.annotation_file=annotations/mpeg-oiv6-detection-coco.json \
                    ++dataset.config.dataset_name=mpeg-oiv6-detection \
                    ++evaluator.type=OIC-EVAL

# MPEGOIV6 - Segmentation with Mask RCNN
python ${ENTRY_CMD} --config-name=eval_example.yaml \
                    ++pipeline.type=unfold \
                    ++pipeline.conformance.save_conformance_files=True \
                    ++pipeline.conformance.subsample_ratio=9 \
                    ++vision_model.arch=mask_rcnn_X_101_32x8d_FPN_3x \
                    ++dataset.type=Detectron2Dataset \
                    ++dataset.datacatalog=MPEGOIV6 \
                    ++dataset.config.root=${MPEG_OIV6_SRC} \
                    ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco.json \
                    ++dataset.config.dataset_name=mpeg-oiv6-segmentation \
                    ++evaluator.type=OIC-EVAL

# SFU - Segmentation with Faster RCNN
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
    python ${ENTRY_CMD} --config-name=eval_example.yaml \
                        ++pipeline.type=fold \
                        ++pipeline.conformance.save_conformance_files=True \
                        ++pipeline.conformance.subsample_ratio=9 \
                        ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
                        ++dataset.type=Detectron2Dataset \
                        ++dataset.datacatalog=SFUHW \
                        ++dataset.config.root=${SFU_HW_SRC}/${SEQ} \
                        ++dataset.config.annotation_file=annotations/${SEQ}.json \
                        ++dataset.config.dataset_name=sfu-hw-${SEQ} \
                        ++evaluator.type=COCO-EVAL
done

# TVD - Object Tracking with JDE
for SEQ in \
            'TVD-01' \
            'TVD-02' \
            'TVD-03'
do
    python ${ENTRY_CMD} --config-name=eval_example.yaml \
                        ++pipeline.type=fold \
                        ++vision_model.arch=jde_1088x608 \
                        ++vision_model.jde_1088x608.splits="[74, 61, 36]" \
                        ++dataset.type=TrackingDataset \
                        ++dataset.settings.patch_size="[608, 1088]" \
                        ++dataset.datacatalog=MPEGTVDTRACKING \
                        ++dataset.config.root=${TVD_SRC}/${SEQ} \
                        ++dataset.config.imgs_folder=img1 \
                        ++dataset.config.annotation_file=gt/gt.txt \
                        ++dataset.config.dataset_name=mpeg-tracking-${SEQ} \
                        ++evaluator.type=MOT-TVD-EVAL
done

# HIEVE - Object Tracking with JDE
for SEQ in \
            '13' \
            '16' \
            '2' \
            '17' \
            '18'
do
    python ${ENTRY_CMD} --config-name=eval_example.yaml \
                        ++pipeline.type=fold \
                        ++pipeline.conformance.save_conformance_files=True \
                        ++pipeline.conformance.subsample_ratio=90 \
                        ++vision_model.arch=jde_1088x608 \
                        ++vision_model.jde_1088x608.splits="[105, 90, 75]" \
                        ++dataset.type=TrackingDataset \
                        ++dataset.settings.patch_size="[608, 1088]" \
                        ++dataset.datacatalog=MPEGHIEVE \
                        ++dataset.config.root=${HIEVE_SRC}/${SEQ} \
                        ++dataset.config.imgs_folder=img1 \
                        ++dataset.config.annotation_file=gt/gt.txt \
                        ++dataset.config.dataset_name=mpeg-hieve-${SEQ} \
                        ++evaluator.type=MOT-HIEVE-EVAL
done
