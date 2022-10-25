#!/usr/bin/env bash
# runs all the bash scripts in this folder for testing the cli, sequentially

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DEFAULT_CODEC="${SCRIPT_DIR}/../../examples/models/bmshj2018-factorized"
VTM=""

#parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_

OPTIONS: [-v|--vtm folder containing VTM software (root), default=""]
         [-c|--custom_codec folder containing compression model for test, default=""]
_EOF_
        exit;
        ;;
        -v|--vtm) shift; VTM="$1"; shift; ;;
        -c|--custom_codec) shift; CODEC="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

if [ "$VTM" = "" ]; then
    echo "enter path for vtm software with -v|--vtm";
    exit;
fi

if [ "${CODEC}" = "" ]; then
    echo "using default: bmshj2018-factorized";
    CODEC=${DEFAULT_CODEC}
fi

bash ${SCRIPT_DIR}/01_auto_import_mock.bash
bash ${SCRIPT_DIR}/02_info_list.bash
bash ${SCRIPT_DIR}/03_download_register_dummy_deregister.bash
bash ${SCRIPT_DIR}/04_vtm.bash ${VTM}
bash ${SCRIPT_DIR}/05_detectron2_eval_vtm.bash ${VTM}
bash ${SCRIPT_DIR}/06_detectron2_eval_custom.bash
bash ${SCRIPT_DIR}/07_detectron2_eval_compressai.bash
bash ${SCRIPT_DIR}/08_plot_csv.bash
bash ${SCRIPT_DIR}/09_plot_img.bash
bash ${SCRIPT_DIR}/10_detectron2_eval_seg.bash
bash ${SCRIPT_DIR}/11_detectron2_eval_no_compress.bash
bash ${SCRIPT_DIR}/12_metrics_eval_compressai.bash
bash ${SCRIPT_DIR}/13_detectron2_eval_compressai.bash

echo "DONE, deleting temporary files"
rm -r /tmp/compressai-vision
