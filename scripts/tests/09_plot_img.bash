#!/usr/bin/env bash
# tests ploting results from detection output files.
# outputs out.png

echo
echo 09
echo

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

compressai-vision plot --dirs="${SCRIPT_DIR}/../../examples/data/interdigital/vtm_scale_100" \
--symbols=x--r --names=vtm --eval=0.792
#,--r --show-baseline=100
