#!/usr/bin/env bash
set -eu

FCM_TESTDATA=""
INNER_CODEC_PATH=""
OUTPUT_DIR=""
EXPERIMENT="test"
DEVICE="cpu"
QP=42
SEQ=""
PIPELINE_PARAMS=""
PIPELINE="split"

export DNNL_MAX_CPU_ISA=AVX2
export DEVICE=${DEVICE}

#parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_
Installs CompressAI-Vision and its dependencies within a virtual environment.
Before running, create a virtual env, i.e.:
$ python3 -m venv venv
$ source venv/bin/activate

RUN OPTIONS:
                [-t|--testdata) path to the test dataset, e.g. path/to/fcm_testdata/, default=""]
                [-p|--pipeline) type of pipeline, split/remote, default="split"]
                [-i|--inner_codec) path to core codec, e.g. /path/to/VTM_repo (that contains subfolders bin/ cfg/ ...), default=""]
                [-o|--output_dir) root of output folder for logs and artifacts, default=""]
                [-e|--exp_name) name of the experiments to locate/label outputs, default="test"]
                [-d|--device) cuda or cpu, default="cpu"]
                [-q|--qp) quality level, depends on the inner codec, default=42]
                [-s|--seq_name) sequence name as used in testdata root folder. E.g., "Traffic_2560x1600_30_val" in sfu_hw_obj, default="42"]
                [-x|--extra_params) additional parameters to override default configs (pipeline/codec/evaluation...), default=""]
EXAMPLE         [bash install_models.sh -t /patht/to/testdata -i /path/to/vtm -o ./here -e test1 -d cuda -q 32 -s Traffic_2560x1600_30_val]
_EOF_
            exit;
            ;;
        -t|--testdata) shift; FCM_TESTDATA="$1"; shift; ;;
        -p|--pipeline) shift; PIPELINE="$1"; shift; ;;
        -i|--inner_codec) shift; INNER_CODEC_PATH="$1"; shift; ;;
        -o|--output_dir) shift; OUTPUT_DIR="$1"; shift; ;;
        -e|--exp_name) shift; EXPERIMENT="$1"; shift; ;;
        -d|--device) shift; DEVICE="$1"; shift; ;;
        -q|--qp) shift; QP="$1"; shift; ;;
        -s|--seq_name) shift; SEQ="$1"; shift; ;;
        -x|--extra_params) shift; PIPELINE_PARAMS="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

DATASET_SRC="${FCM_TESTDATA}/tvd_tracking"

CONF_NAME="eval_split_inference_example.yaml"
if [[ ${PIPELINE} == "remote" ]]; then
  CONF_NAME="eval_remote_inference_example.yaml"
fi


declare -A intra_period_dict
declare -A fr_dict

intra_period_dict["TVD-01"]=64
fr_dict["TVD-01"]=50

intra_period_dict["TVD-02"]=64
fr_dict["TVD-02"]=50

intra_period_dict["TVD-03"]=64
fr_dict["TVD-03"]=50

INTRA_PERIOD=${intra_period_dict[${SEQ}]}
FRAME_RATE=${fr_dict[${SEQ}]}

echo "============================== RUNNING COMPRESSAI-VISION EVAL== =================================="
echo "Pipeline Type:      " ${PIPELINE} " Video"
echo "Datatset location:  " ${FCM_TESTDATA}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "vtm"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Input sequence:     " ${SEQ}
echo "Seq. Framerate:     " ${FRAME_RATE}
echo "QP for Inner Codec: " ${QP}
echo "Intra Period for Inner Codec: "${INTRA_PERIOD}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="

compressai-${PIPELINE}-inference --config-name=${CONF_NAME} \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
	++vision_model.arch=jde_1088x608 \
        ++vision_model.jde_1088x608.splits="[36, 61, 74]" \
        ++dataset.type=TrackingDataset \
        ++dataset.datacatalog=MPEGTVDTRACKING \
	++dataset.settings.patch_size="[608, 1088]" \
        ++dataset.config.root=${DATASET_SRC}/${SEQ} \
        ++dataset.config.imgs_folder=img1 \
       	++dataset.config.annotation_file=gt/gt.txt \
        ++dataset.config.dataset_name=mpeg-${SEQ} \
        ++evaluator.type=MOT-TVD-EVAL \
        ++codec.experiment=${EXPERIMENT} \
	codec=vtm.yaml \
        ++codec.encoder_config.intra_period=${INTRA_PERIOD} \
        ++codec.encoder_config.parallel_encoding=True \
        ++codec.encoder_config.qp=${QP} \
        ++codec.codec_paths.encoder_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.codec_paths.decoder_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.codec_paths.parcat_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
        ++codec.codec_paths.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_lowdelay_vtm.cfg' \
        ++codec.eval_encode='bitrate' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
        ${PIPELINE_PARAMS} \
        