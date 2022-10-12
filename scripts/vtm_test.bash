#!/bin/bash
#
# Stand-alone VTM test 
#
##download & compile VTM:
# RUN wget https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/archive/VTM-12.0/VVCSoftware_VTM-VTM-12.0.tar.gz
# RUN tar xvf VVCSoftware_VTM-VTM-12.0.tar.gz
# WORKDIR /root/vtm/VVCSoftware_VTM-VTM-12.0/build
# RUN cmake .. -DCMAKE_BUILD_TYPE=Release
# RUN make -j
#
inp="1.png" # use ffprobe to check out the dims
# WARNING: be sure that image dims are divisible by 2
width="768"
height="512"
vtm_base="/home/sampsa/silo/interdigital/VVCSoftware_VTM" # TODO: EDIT
exe_encode=$vtm_base"/bin/EncoderAppStatic"
exe_decode=$vtm_base"/bin/DecoderAppStatic"
cfg="/home/sampsa/silo/interdigital/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"
#
echo
echo CLEANING OLD FILES
echo
rm -f tmp.png inp.yuv bytes.bin out.yuv rec.png rec2.png
# ffmpeg -y -i $inp -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" tmp.png
cp -f $inp tmp.png
echo 
echo TO YUV
echo
com="ffmpeg -y -i tmp.png -f rawvideo -pix_fmt yuv420p -dst_range 1 inp.yuv"
echo $com
$com
echo
echo ENCODE
echo
com=$exe_encode" -c $cfg -i inp.yuv -b bytes.bin -o out.yuv -fr 1 -f 1 -wdt $width -hgt $height -q 47 --ConformanceWindowMode=1 --InternalBitDepth=10"
echo $com
$com
# exit 2
rm -f out.yuv
echo
echo DECODE
echo
com=$exe_decode" -b bytes.bin -o out.yuv"
echo $com
$com
# exit 2
echo 
echo TO PNG
echo 
com="ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s "$width"x"$height" -src_range 1 -i out.yuv -frames 1  -pix_fmt rgb24 rec.png"
echo $com
$com
# ffmpeg -y -i rec.png -vf "crop="$width":"$height rec2.png
