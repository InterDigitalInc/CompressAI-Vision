#!/usr/bin/env bash
#
# Use plot function to output combined csv tables collecting results for the different
# qps from json files

echo
echo 08
echo

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

compressai-vision plot --csv --dirs="${SCRIPT_DIR}/../../examples/data/interdigital/vtm_scale_100"

