#! /usr/bin/env bash
output_dir=$1
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
find ${output_dir} -name "*.bin" -o -name "*.dump" | tar -cf cfp_bitstream_${TIMESTAMP}.tar.gz -T -