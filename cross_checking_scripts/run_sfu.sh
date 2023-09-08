
# change the paths accrodingly
#####################################################
INPUT_DIR="/data/datasets/MPEG-FCVCM/vcm_testdata"
OUTPUT_DIR="./cfp_run"
EXPERIMENT="_sfu_cfp_test"
DEVICE="cpu"
#####################################################
MAX_PARALLEL=6 # total number of jobs = 84

QPS=`echo "5 6 7 8 9 10"`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

parallel_run () {
	sem -j $MAX_PARALLEL bash "$1"
}

export -f parallel_run

for SEQ in \
            'Traffic_2560x1600_30_val' \
            'Kimono_1920x1080_24_val' \
            'ParkScene_1920x1080_24_val' \
            'Cactus_1920x1080_50_val' \
            'BasketballDrive_1920x1080_50_val' \
            'BQTerrace_1920x1080_60_val' \
            'BasketballDrill_832x480_50_val' \
            'BQMall_832x480_60_val' \
            'PartyScene_832x480_50_val' \
            'RaceHorses_832x480_30_val' \
            'BasketballPass_416x240_50_val' \
            'BQSquare_416x240_60_val' \
            'BlowingBubbles_416x240_50_val' \
            'RaceHorses_416x240_30_val'
do
    for QP in ${QPS}
    do
        echo Input Dir: ${INPUT_DIR}, Output Dir: ${OUTPUT_DIR}, Exp Name: ${EXPERIMENT}, Device: ${DEVICE}, QP: ${QP}, SEQ Name: ${SEQ}
        parallel_run "${SCRIPT_DIR}/mpeg_cfp_sfu.sh ${INPUT_DIR} ${OUTPUT_DIR} ${EXPERIMENT} ${DEVICE} ${QP} ${SEQ}"
    done
done

sem --wait
python ${SCRIPT_DIR}/../utils/mpeg_template_format.py --dataset SFU --result_path ${OUTPUT_DIR}/split-inference-video/cfp_codec${EXPERIMENT}/SFUHW/

