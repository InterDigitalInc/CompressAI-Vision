#!/usr/bin/env bash

set -eu

OUTPUT_DIR="/mnt/wekamount/scratch_fcvcm/eimran/runs"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENTRY_CMD="compressai-vision-eval"

VCM_TESTDATA="/data/datasets/MPEG-FCVCM/vcm_testdata"
MPEG_OIV6_SRC="${VCM_TESTDATA}/mpeg-oiv6"


for qp in {5..10}
do
    ${ENTRY_CMD} --config-name=eval_cfp_codec.yaml \
            ++paths._runs_root="/mnt/wekamount/scratch_fcvcm/eimran/runs" \
            ++pipeline.type=image \
            ++pipeline.conformance.save_conformance_files=True \
            ++pipeline.conformance.subsample_ratio=9 \
            ++codec.encoder_config.qp=${qp} \
            ++codec.eval_encode='bpp' \
            ++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
            ++dataset.type=Detectron2Dataset \
            ++dataset.datacatalog=MPEGOIV6 \
            ++dataset.config.root=${MPEG_OIV6_SRC} \
            ++dataset.config.annotation_file=annotations/mpeg-oiv6-detection-coco.json \
            ++dataset.config.dataset_name=mpeg-oiv6-detection \
            ++evaluator.type=OIC-EVAL

    ${ENTRY_CMD} --config-name=eval_cfp_codec.yaml \
        ++paths._runs_root="/mnt/wekamount/scratch_fcvcm/eimran/runs" \
        ++pipeline.type=image \
        ++pipeline.conformance.save_conformance_files=True \
        ++pipeline.conformance.subsample_ratio=9 \
        ++codec.encoder_config.qp=${qp} \
        ++codec.eval_encode='bpp' \
        ++vision_model.arch=mask_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=MPEGOIV6 \
        ++dataset.config.root=${MPEG_OIV6_SRC} \
        ++dataset.config.annotation_file=annotations/mpeg-oiv6-segmentation-coco.json \
        ++dataset.config.dataset_name=mpeg-oiv6-segmentation \
        ++evaluator.type=OIC-EVAL
done


python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset OIV6 --result_path ${OUTPUT_DIR}/split-inference-image/cfp_codec/MPEGOIV6/