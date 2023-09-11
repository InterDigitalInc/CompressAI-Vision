#! /usr/bin/env bash
output_dir=$1
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

cd ${output_dir}

find . -name "*.bin" -o -name "*.dump" | tar -cf cfp_bitstream_${TIMESTAMP}.tar.gz -T -

echo "bitstreams tar file saved at..."
echo ${output_dir}/cfp_bitstream_${TIMESTAMP}.tar.gz
