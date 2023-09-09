# change the paths accrodingly
#####################################################
INPUT_DIR="/data/datasets/MPEG-FCVCM/vcm_testdata"
OUTPUT_DIR="./cfp_run"
EXPERIMENT="_oiv6_segm_cfp_test"
DEVICE="cpu"
MAX_PARALLEL=6 # total number of job = 6
#####################################################
QPS=`echo "7 8 9 10 11 12"`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

parallel_run () {
	sem -j $MAX_PARALLEL bash "$1"
}

export -f parallel_run


# run seg
for QP in ${QPS}
do
    echo Input Dir: ${INPUT_DIR}, Output Dir: ${OUTPUT_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}
    parallel_run "${SCRIPT_DIR}/mpeg_cfp_oiv6_segmentation.sh ${INPUT_DIR} ${OUTPUT_DIR} ${EXPERIMENT} ${DEVICE} ${QP}"
done

sem --wait
python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset OIV6 --result_path ${OUTPUT_DIR}/split-inference-image/cfp_codec${EXPERIMENT}/MPEGOIV6/mpeg-oiv6-segmentation
