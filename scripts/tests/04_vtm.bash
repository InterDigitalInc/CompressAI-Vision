#!/usr/bin/env bash
#
# This tests vtm encode/decode on the mpeg dataset
# It requires having imported and registered mpeg-vcm-detection
# i.e. run -1_auto_import_mock.bash
set -e

echo
echo 04
echo

VTM_BASE=""

#parse args
VTM_BASE="$1"

if [ "$VTM_BASE" = "" ]; then
    echo "arg missing: root folder of VTM software";
    exit 1;
fi

VTM_DIR=$VTM_BASE"/bin"
VTM_CFG=$VTM_BASE"/cfg/encoder_intra_vtm.cfg"

compressai-vision vtm --y --dataset-name=mpeg-vcm-detection \
--slice=0:2 \
--scale=100 \
--progress=1 \
--qpars=47 \
--vtm_cache=/tmp/compressai-vision \
--vtm_dir=$VTM_DIR \
--vtm_cfg=$VTM_CFG \
--output=vtm_test.json