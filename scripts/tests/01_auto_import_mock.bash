#!/usr/bin/env bash
# mocks import and registration of supported mpeg vcm datasets
set -e

echo
echo 01
echo

compressai-vision mpeg-vcm-auto-import --mock
