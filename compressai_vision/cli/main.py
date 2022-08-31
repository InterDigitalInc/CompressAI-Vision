"""cli.py : Command-line interface tools for compressai-vision

* Copyright: 2022 InterDigital
* Authors  : Sampsa Riikonen
* Date     : 2022
* Version  : 0.1

This file is part of compressai-vision libraru
"""
import argparse

# import configparser  # https://docs.python.org/3/library/configparser.html
# import logging

# from compressai_vision import constant
# from compressai_vision.local import AppLocalDir
from compressai_vision.tools import quickLog
import logging

def process_cl_args():
    # def str2bool(v):
    #     return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(
        usage="""
compressai-vision [options] command

for an all-automagic-no-brainer, please try "compressai-nokia-auto-import help"
that substitutes for download, nokia_convert and register commands

commands & parameters:

    --y                 non-interactive

    download            download image set _and_ register it to fiftyone.
        --name          name of the dataset.  Default: "open-images-v6". --lists
        lst files that define the subset of images to download. --split
        typically "train" or "validation".  Default: "validation". example:
        (donwloads images corresponding to nokia's detection & segmentation test
        sets)
            compressai-vision download \\
            --lists=detection_validation_input_5k.lst,segmentation_validation_input_5k.lst
            \\ --name=open-image-v6 --split=validation

    list                list all image sets registered to fiftyone

    deregister          de-register image set from fiftyine
        --name          name of the dataset, for example "open-image-v6" --split
        name of the split, for example "validation"

    dummy               create & register a dummy database with just the first sample
        --name          name of the original database
                        name of the new database will be appended with "-dummy"

    nokia_convert       convert nokia-provided files into proper OpenImageV6
    format & directory structure
        --lists         file listing necessary images, i.e.
                        "detection_validation_input_5k.lst" or
                        "segmentation_validation_input_5k.lst"
        --dir           a base OpenImageV6 dataset dir:
                        images are taken from "[dir]/data/" and masks from
                        "[dir]/labels/masks/"
        --target_dir    target directory directory for the converted format
        --label         image-level labels, i.e.
                        "detection_validation_labels_5k.csv" or
                        "segmentation_validation_labels_5k.csv"
        --bbox          bbox data, i.e.
                        "detection_validation_5k_bbox.csv" or
                        "segmentation_validation_bbox_5k.csv"
        --mask          segmentation masks,
        i.e."segmentation_validation_masks_5k.csv" example:
            compressai-vision nokia-convert
            --lists=detection_validation_input_5k.lst \\
            --bbox=detection_validation_5k_bbox.csv
            --label=detection_validation_labels_5k.csv \\
            --dir=~/fiftyone/open-image-v6/validation --target_dir=/path/to/dir

    register            register image set to fiftyone from local dir
        --name          database registered name --lists         lst files that
        define the subset of images to register --dir           source directory
        --type          fiftyone.types name.  Default: "OpenImagesV6Dataset"
        example:        (register converted nokia files into fiftyone)
            compressai-vision register --name=nokia-import
            --dir=~/somedir/nokia-import

    detectron2_eval     evaluate model with detectron2 using OpenImageV6
    evaluation protocol
                        optionally with no (de)compression or with compressai or
                        vtm
        --name          name of the fiftyone registered database --model
        name of the detectron2 model
                        for example:
                        COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        --output        outputfile, default: compressai-vision.json --compressai
        compressai model (optional), for example: bmshj2018_factorized --vtm
        use vtm model (optional) --qpars         quality parameters to be used
        with either compressai or vtm WARNING/TODO : currently works only for
        detections (not for segmentations) 
        
        --vtm_dir       specify path to directory "VVCSoftware_VTM/bin", i.e. to the
                        directory where there are executables "EncoderAppStatic" and 
                        "DecoderAppStatic".  If not specified, tries to use the
                        environmental variable VTM_DIR
        
        --ffmpeg        specify ffmpeg command.  If not specified, uses "ffmpeg".
        
        --vtm_cfg       path to vtm config file.  If not specified uses an internal
                        default file.

        --debug         debug verbosity for the CompressAI & VTM

        example 1 (calculate mAP):
            compressai-vision detectron2_eval --y --name=nokia-detection \\
            --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        example 2 (calculate mAP=mAP(bpp)):
            compressai-vision detectron2_eval --y --name=nokia-detection \\
            --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \\
            --compressai=bmshj2018_factorized --qpars=1,2,3,4,5,6,7,8

    NOTE: your normal workflow would be: download, nokia_convert, register,
    detectron2_eval the first three steps can be dealt with the
    compress-nokia-auto-import command

    """
    )
    # parser.register('type','bool',str2bool)  # this works only in theory..
    parser.add_argument("command", action="store", type=str, help="mandatory command")

    parser.add_argument(
        "--name",
        action="store",
        type=str,
        required=False,
        default=None,
        help="name of the dataset",
    )
    parser.add_argument(
        "--lists",
        action="store",
        type=str,
        required=False,
        default=None,
        help="comma-separated list of list files",
    )
    parser.add_argument(
        "--split",
        action="store",
        type=str,
        required=False,
        default="validation",
        help="database sub-name, say, 'train' or 'validation'",
    )

    parser.add_argument(
        "--dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="source directory, depends on command",
    )
    parser.add_argument(
        "--target_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="target directory for nokia_convert",
    )
    parser.add_argument(
        "--label",
        action="store",
        type=str,
        required=False,
        default=None,
        help="nokia-formatted image-level labels",
    )
    parser.add_argument(
        "--bbox",
        action="store",
        type=str,
        required=False,
        default=None,
        help="nokia-formatted bbox data",
    )
    parser.add_argument(
        "--mask",
        action="store",
        type=str,
        required=False,
        default=None,
        help="nokia-formatted segmask data",
    )

    parser.add_argument(
        "--type",
        action="store",
        type=str,
        required=False,
        default="OpenImagesV6Dataset",
        help="image set type to be imported",
    )

    parser.add_argument(
        "--proto",
        action="store",
        type=str,
        required=False,
        default=None,
        help="evaluation protocol",
    )
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        required=False,
        default=None,
        help="detectron2 model",
    )
    parser.add_argument(
        "--output",
        action="store",
        type=str,
        required=False,
        default="compressai-vision.json",
        help="results output file",
    )

    parser.add_argument(
        "--compressai",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use compressai model",
    )
    parser.add_argument("--vtm", action="store_true", default=False)
    parser.add_argument(
        "--qpars",
        action="store",
        type=str,
        required=False,
        default=None,
        help="quality parameters for compressai model or vtm",
    )
    parser.add_argument(
        "--vtm_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="path to directory with executables EncoderAppStatic & DecoderAppStatic"
    )
    parser.add_argument(
        "--ffmpeg",
        action="store",
        type=str,
        required=False,
        default="ffmpeg",
        help="ffmpeg command"
    )
    parser.add_argument(
        "--vtm_cfg",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm config file"
    )
    

    parser.add_argument("--debug", action="store_true", default=False, help="debug verbosity")
    parser.add_argument("--y", action="store_true", default=False, help="non-interactive run")

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args


def main():
    parsed, unparsed = process_cl_args()

    for weird in unparsed:
        print("invalid argument", weird)
        raise SystemExit(2)

    # setting loglevels manually
    # should only be done in tests:
    """
    logger = logging.getLogger("name.space")
    confLogger(logger, logging.INFO)
    """
    if parsed.debug:
        loglev=logging.DEBUG
    else:
        loglev=logging.INFO
    quickLog("CompressAIEncoderDecoder", loglev)
    quickLog("VTMEncoderDecoder", loglev)

    # some command filtering here
    if parsed.command in [
        "download",
        "list",
        "dummy",
        "deregister",
        "nokia_convert",
        "register",
        "detectron2_eval",
        "load_eval",
    ]:
        from compressai_vision import cli

        # print("command is", parsed.command)
        func = getattr(cli, parsed.command)
        func(parsed)
    else:
        print("unknown command", parsed.command)
        raise SystemExit(2)
    # some ideas on how to handle config files & default values
    #
    # this directory is ~/.skeleton/some_data/ :
    # init default data with yaml constant string
    """
    some_data_dir = AppLocalDir("some_data")
    if (not some_data_dir.has("some.yml")) or parsed.reset:
        with open(some_data_dir.getFile("some.yml"), "w") as f:
            f.write(constant.SOME)
    """

if __name__ == "__main__":
    main()
