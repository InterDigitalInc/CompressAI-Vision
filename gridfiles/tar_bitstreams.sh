#! /usr/bin/env bash
output_dir=$1
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

cd ${output_dir}

find . -name "*.bin"  | tar -cf fcvcm-cfp-proposal16_bitstreams_and_decoder_${TIMESTAMP}.tar.gz -T -
find . -name "*.dump" | tar -cf fcvcm-cfp-proposal16_feature_dumps_${TIMESTAMP}.tar.gz -T -

echo "tar files saved at..."
echo ${output_dir}/fcvcm-cfp-proposal16_bitstreams_and_decoder_${TIMESTAMP}.tar.gz
echo ${output_dir}/fcvcm-cfp-proposal16_feature_dumps_${TIMESTAMP}.tar.gz
