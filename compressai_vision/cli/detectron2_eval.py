# Copyright (c) 2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""cli detectron2_eval functionality
"""
import copy
import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Dict

from torch import Tensor


def rename_key(key: str) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict

def add_subparser(subparsers, parents=[]):
    subparser = subparsers.add_parser("detectron2-eval", parents=parents)
    subparser.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset",
    )
    subparser.add_argument(
        "--gt-field",
        action="store",
        type=str,
        required=False,
        default="detections",
        help="name of the ground truth field in the dataset.  Default: detections",
    )
    subparser.add_argument(
        "--model",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of Detectron2 config model",
    )
    subparser.add_argument(
        "--output",
        action="store",
        type=str,
        required=False,
        default="compressai-vision.json",
        help="outputfile, default: compressai-vision.json",
    )
    """TODO: not only oiv6 protocol, but coco etc.
    subparser.add_argument(
        "--proto",
        action="store",
        type=str,
        required=False,
        default=None,
        help="evaluation protocol",
    )
    """
    subparser.add_argument(
        "--compressai-model-name",
        action="store",
        type=str,
        required=False,
        default=None,
        help="name of an existing model in compressai (e.g. 'cheng2020-attn')",
    )
    subparser.add_argument(
        "--compression-model-path",
        action="store",
        type=str,
        required=False,
        default=None,
        help="a path to a directory containing model.py for custom development model",
    )
    subparser.add_argument(
        "--compression-model-checkpoint",
        action="store",
        type=str,
        nargs="*",
        required=False,
        default=None,
        help="path to a compression model checkpoint(s)",
    )
    subparser.add_argument("--vtm", action="store_true", default=False)
    subparser.add_argument(
        "--vtm_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="path to directory with executables EncoderAppStatic & DecoderAppStatic",
    )
    subparser.add_argument(
        "--vtm_cfg",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm config file",
    )
    subparser.add_argument(
        "--vtm_cache",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory to cache vtm bitstreams",
    )
    subparser.add_argument(
        "--qpars",
        action="store",
        type=str,
        required=False,
        default=None,
        help="quality parameters for compressai model or vtm",
    )
    subparser.add_argument(
        "--scale",
        action="store",
        type=int,
        required=False,
        default=100,
        help="image scaling as per VCM working group docs",
    )

    subparser.add_argument(
        "--ffmpeg",
        action="store",
        type=str,
        required=False,
        default="ffmpeg",
        help="ffmpeg command",
    )
    subparser.add_argument(
        "--slice",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use a dataset slice instead of the complete dataset",
    )
    # subparser.add_argument("--debug", action="store_true", default=False) # not here
    subparser.add_argument(
        "--progressbar",
        action="store_true",
        default=False,
        help="show fancy progressbar",
    )
    subparser.add_argument(
        "--progress",
        action="store",
        type=int,
        required=False,
        default=1,
        help="Print progress this often",
    )
    subparser.add_argument(
        "--eval-method",
        action="store",
        type=str,
        required=False,
        default="open-images",
        help="Evaluation method/protocol: open-images or coco.  Default: open-images",
    )
    return subparser


def main(p):  # noqa: C901

    # check that only one is defined
    defined_codec = ""
    for codec in [p.compressai_model_name, p.vtm, p.compression_model_path]:
        if codec:
            if defined_codec:  # second match!
                raise AssertionError(
                    "please define only one of the following: compressai_model_name, vtm or compression_model_path"
                )
            defined_codec = codec
    assert p.dataset_name is not None, "please provide dataset name"
    assert p.model is not None, "provide Detectron2 model name"

    # fiftyone
    print("importing fiftyone")
    import fiftyone as fo

    from compressai_vision import patch  # dataset.clone needs this

    print("fiftyone imported")

    from compressai_vision.constant import vf_per_scale

    # compressai_vision
    from compressai_vision.evaluation.fo import (  # annex predictions from
        annexPredictions,
    )
    from compressai_vision.evaluation.pipeline import (
        CompressAIEncoderDecoder,
        VTMEncoderDecoder,
    )
    from compressai_vision.tools import getDataFile


    try:
        dataset = fo.load_dataset(p.dataset_name)
    except ValueError:
        print("FATAL: no such registered dataset", p.dataset_name)
        return

    if p.slice is not None:
        print("WARNING: using a dataset slice instead of full dataset")
        print("SURE YOU WANT THIS?")
        # say, 0:100
        nums = p.slice.split(":")
        if len(nums) < 2:
            print("invalid slicing: use normal python slicing, say, 0:100")
            return
        try:
            fr = int(nums[0])
            to = int(nums[1])
        except ValueError:
            print("invalid slicing: use normal python slicing, say, 0:100")
            return
        assert to > fr, "invalid slicing: use normal python slicing, say, 0:100"
        dataset = dataset[fr:to]



    if (
        (p.compressai_model_name is None)
        == p.vtm
        == (p.compression_model_path is None)
        == False
    ):
        compression = False
        # no (de)compression, just eval
        assert (
            qpars is None
        ), "you have provided quality pars but not a (de)compress model"



    if defined_codec != p.compression_model_path:
        # check quality parameter list
        assert p.qpars is not None, "need to provide integer quality parameters"
        try:
            qpars = [int(i) for i in p.qpars.split(",")]
        except Exception as e:
            print("problems with your quality parameter list")
            raise e

    else:
        #if model checkpoints provided, loop over the checkpoints
        qpars = p.compression_model_checkpoint

    if p.compressai_model_name is not None:  # compression from compressai zoo
        from compressai.zoo import image_models as pretrained_models

        # compressai_model = getattr(compressai.zoo, "bmshj2018_factorized")
        compression_model = pretrained_models[p.compressai_model_name]

    # compressai.zoo. getattr(
    #         compressai.zoo, p.compressai_model_name
    #     )  # a function that returns a model instance or just a class

    elif (
        p.compression_model_path is not None
    ):  # compression from a custcom compression model
    # TODO (fracape) why not asking for the full file path?
        model_file = Path(p.compression_model_path) / "model.py"
        if model_file.is_file():
            import importlib.util

            try:
                spec = importlib.util.spec_from_file_location("module", model_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                print(
                    "loading model from directory",
                    p.compression_model_path,
                    "failed with",
                    e,
                )
                return
            else:
                assert hasattr(
                    module, "getModel"
                ), "your module is missing getModel function"
                compression_model = (
                    module.getModel
                )  # a function that returns a model instance or just a class
                print("loaded custom model.py")
            assert p.compression_model_checkpoint is not None
            # for checkpoint_file in p.compression_model_checkpoint:
            #     try:
            #         _ = Path(checkpoint_file).resolve(strict=True)
            #     except FileNotFoundError:
            #         # doesn't exist
        else:
            raise FileNotFoundError(f"No model.py in {p.compression_model_path}")


    elif p.vtm:  # setup VTM
        if p.vtm_dir is None:
            try:
                vtm_dir = os.environ["VTM_DIR"]
            except KeyError as e:
                print("please define --vtm_dir or set environmental variable VTM_DIR")
                # raise e
                return
        else:
            vtm_dir = p.vtm_dir
        vtm_dir = os.path.expanduser(vtm_dir)
        if p.vtm_cfg is None:
            vtm_cfg = getDataFile("encoder_intra_vtm_1.cfg")
            print("WARNING: using VTM default config file", vtm_cfg)
        else:
            vtm_cfg = p.vtm_cfg
        vtm_cfg = os.path.expanduser(
            vtm_cfg
        )  # some more systematic way of doing these..
        assert os.path.isfile(vtm_cfg), "vtm config file not found"
        # try both filenames..
        vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStatic")
        if not os.path.isfile(vtm_encoder_app):
            vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStaticd")
        if not os.path.isfile(vtm_encoder_app):
            print("FATAL: can't find EncoderAppStatic(d) in", vtm_dir)
        # try both filenames..
        vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStatic")
        if not os.path.isfile(vtm_decoder_app):
            vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStaticd")
        if not os.path.isfile(vtm_decoder_app):
            print("FATAL: can't find DecoderAppStatic(d) in", vtm_dir)

    # *** CHOOSE COMPRESSION SCHEME OK ***

    if p.scale is not None:
        assert p.scale in vf_per_scale.keys(), "invalid scale value"

    # *** Detectron imports ***
    # Some basic setup:
    # Setup detectron2 logger
    # import detectron2
    import torch

    from detectron2.utils.logger import setup_logger

    setup_logger()

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog  # , DatasetCatalog
    from detectron2.engine import DefaultPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    model_name = p.model

    # cfg encapsulates the model architecture & weights, also threshold parameter, metadata, etc.
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    # load config from a file:
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    # DO NOT TOUCH THRESHOLD WHEN DOING EVALUATION:
    # too big a threshold will cut the smallest values
    # & affect the precision(recall) curves & evaluation results
    # the default value is 0.05
    # value of 0.01 saturates the results (they don't change at lower values)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # get weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    # print("expected input colorspace:", cfg.INPUT.FORMAT)
    # print("loaded datasets:", cfg.DATASETS)
    model_dataset = cfg.DATASETS.TRAIN[0]
    # print("model was trained with", model_dataset)
    model_meta = MetadataCatalog.get(model_dataset)

    predictor_field = "detectron-predictions"
    ## instead, create a unique identifier for the field
    ## in this run: this way parallel runs dont overwrite
    ## each other's field
    ## as the database is the same for each running instance/process
    # ui=uuid.uuid1().hex # 'e84c73f029ee11ed9d19297752f91acd'
    # predictor_field = "detectron-"+ui
    # predictor_field = "detectron-{0:%Y-%m-%d-%H-%M-%S-%f}".format(
    #    datetime.datetime.now()
    # )
    # even better idea: create a temporarily cloned database
    try:
        username = os.environ["USER"]
    except KeyError:
        username = "nouser"
    tmp_name0 = p.dataset_name + "-{0:%Y-%m-%d-%H-%M-%S-%f}".format(
        datetime.datetime.now()
    )

    tmp_name = "detectron-run-{username}-{tmp_name0}".format(
        username=username, tmp_name0=tmp_name0
    )

    # eval_method="open-images" # could be changeable in the future..
    # eval_method="coco"
    eval_method=p.eval_method

    print()
    print("Using dataset          :", p.dataset_name)
    print("Dataset tmp clone      :", tmp_name)
    print("Image scaling          :", p.scale)
    if p.slice is not None:  # woops.. can't use slicing
        print("WARNING: Using slice   :", str(fr) + ":" + str(to))
    print("Number of samples      :", len(dataset))
    print("Torch device           :", device)
    print("Detectron2 model       :", model_name)
    print("Model was trained with :", model_dataset)
    if p.compressai_model_name is not None:
        print("Using compressai model :", p.compressai_model_name)
    elif p.compression_model_path is not None:
        print("Using custom model.py from")
        print("                       :", p.compression_model_path)
    elif p.vtm:
        print("Using VTM               ")
        if p.vtm_cache:
            # assert(os.path.isdir(p.vtm_cache)), "no such directory "+p.vtm_cache
            # ..created by the VTMEncoderDecoder class
            print("WARNING: VTM USES CACHE IN", p.vtm_cache)
    else:
        print("** Evaluation without Encoding/Decoding **")
    if p.compression_model_checkpoint:
        print("WARN: using checkpoints")
    if qpars is not None:
        print("Quality parameters     :", qpars)
    print("Ground truth data field name")
    print("                       :", p.gt_field)
    print("Eval. results will be saved to datafield")
    print("                       :", predictor_field)
    print("Evaluation protocol    :", eval_method)


    dataset_ = fo.load_dataset(p.dataset_name)
    if dataset_.get_field(p.gt_field) is None:
        print("FATAL: your dataset does not have requested field '"+p.gt_field+"'")
        print("Dataset info:")
        print(dataset_)
        return

    # print("(if aborted, start again with --resume=%s)" % predictor_field)
    print("Progressbar            :", p.progressbar)
    if p.progressbar and p.progress > 0:
        print("WARNING: progressbar enabled --> disabling normal progress print")
        p.progress = 0
    print("Print progress         :", p.progress)
    print("Output file            :", p.output)
    classes = dataset.distinct("%s.detections.label" % (p.gt_field))
    classes.sort()
    detectron_classes = copy.deepcopy(model_meta.thing_classes)
    detectron_classes.sort()
    print("Peek model classes     :")
    print(detectron_classes[0:5], "...")
    print("Peek dataset classes   :")
    print(classes[0:5], "...")

    if not p.y:
        input("press enter to continue.. ")

    # save metadata about the run into the json file
    metadata = {
        "dataset": p.dataset_name,
        "gt_field": p.gt_field,
        "tmp datasetname": tmp_name,
        "slice": p.slice,
        "model": model_name,
        "codec": defined_codec,
        "qpars": qpars,
    }
    with open(p.output, "w") as f:
        f.write(json.dumps(metadata, indent=2))

    # please see ../monkey.py for problems I encountered when cloning datasets
    # simultaneously with various multiprocesses/batch jobs
    print("cloning dataset", p.dataset_name, "to", tmp_name)
    dataset = dataset.clone(tmp_name)
    dataset.persistent = True
    # fo.core.odm.database.sync_database() # this would've helped? not sure..

    # parameters for dataset.evaluate_detections
    eval_args={
        "gt_field": p.gt_field,
        "method": eval_method,
        "compute_mAP":True
    }
    if eval_method == "open-images":
        if dataset.get_field("positive_labels"):
            eval_args["pos_label_field"] = "positive_labels"
        if dataset.get_field("negative_labels"):
            eval_args["neg_label_field"] = "negative_labels"
        eval_args["expand_pred_hierarchy"] = False
        eval_args["expand_gt_hierarchy"] = False

    print("instantiating Detectron2 predictor")
    predictor = DefaultPredictor(cfg)

    # bpp, mAP values, mAP breakdown per class
    def per_class(results_obj):
        """take fiftyone/openimagev6 results object & spit
        out mAP breakdown as per class
        """
        d = {}
        for class_ in classes:
            d[class_] = results_obj.mAP([class_])
        return d

    xs = []
    ys = []
    maps = []

    # qpars is not None == we perform compression before detectron2
    if qpars is not None:
        # loglev=logging.DEBUG # this now set in main
        # loglev = logging.INFO
        # quickLog("CompressAIEncoderDecoder", loglev)
        # quickLog("VTMEncoderDecoder", loglev)
        for quality in qpars:
            print("\nQUALITY PARAMETER OR CHECKPOINT: ", quality)
            enc_dec = None  # default: no encoding/decoding
            if (
                p.compressai_model_name or p.compression_model_path
            ):  # compressai model, either from the zoo or from a directory
                if p.compression_model_checkpoint is None:
                    net = (
                        compression_model(quality=quality, pretrained=True).eval().to(device)
                    )
                    # e.g. compressai.zoo.bmshj2018_factorized
                    # or a custom model from a file
                else:  # load a checkpoint,

                    net = compression_model()
                    # make sure we load just trained models and pre-trained/ updated entropy parameters
                    try:
                        checkpoint = torch.load(quality)
                        if "network" in checkpoint:
                            state_dict = checkpoint["network"]
                        elif "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]
                        else:
                            state_dict = checkpoint

                        state_dict = load_state_dict(state_dict)
                        # For now, update is forced in case
                        net = net.from_state_dict(state_dict)
                        net.update(force=True)
                        net = net.to(device).eval()
                    except Exception as e:
                        print("\nLoading checkpoint failed!\n")
                        raise e
                enc_dec = CompressAIEncoderDecoder(
                    net, device=device, scale=p.scale, ffmpeg=p.ffmpeg
                )
            elif p.vtm:
                enc_dec = VTMEncoderDecoder(
                    encoderApp=vtm_encoder_app,
                    decoderApp=vtm_decoder_app,
                    ffmpeg=p.ffmpeg,
                    vtm_cfg=vtm_cfg,
                    qp=quality,
                    cache=p.vtm_cache,
                    scale=p.scale,
                    warn=True,
                )
            else:
                raise BaseException("program logic error")

            bpp = annexPredictions(
                predictor=predictor,
                fo_dataset=dataset,
                encoder_decoder=enc_dec,
                gt_field=p.gt_field,
                predictor_field=predictor_field,
                use_pb=p.progressbar,
                use_print=p.progress,
            )

            if bpp is None or bpp < 0:
                print()
                print("Sorry, mAP calculation aborted")
                ##TODO: implement:
                # print("If you want to resume, start again with:")
                # print("--continue", predictor_field)
                print()
                return

            # print("evaluating dataset", dataset.name)
            res = dataset.evaluate_detections(
                predictor_field,
                **eval_args
                #gt_field=p.gt_field,
                #method="open-images",
                #pos_label_field="positive_labels",
                #neg_label_field="negative_labels",
                #expand_pred_hierarchy=False,
                #expand_gt_hierarchy=False,
            )
            xs.append(bpp)
            ys.append(res.mAP())
            maps.append(per_class(res))

    else:  # a pure evaluation without encoding/decoding
        bpp = annexPredictions(
            predictor=predictor,
            fo_dataset=dataset,
            gt_field=p.gt_field,
            predictor_field=predictor_field,
            use_pb=p.progressbar,
            use_print=p.progress,
        )
        res = dataset.evaluate_detections(
            predictor_field,
            **eval_args
            #gt_field=p.gt_field,
            #method="open-images",
            #pos_label_field="positive_labels",
            #neg_label_field="negative_labels",
            #expand_pred_hierarchy=False,
            #expand_gt_hierarchy=False,
        )
        # print(res)
        xs.append(bpp)
        ys.append(res.mAP())
        maps.append(per_class(res))

    # print(">>", metadata)
    metadata["bpp"] = xs
    metadata["map"] = ys
    metadata["map_per_class"] = maps
    with open(p.output, "w") as f:
        f.write(json.dumps(metadata, indent=2))

    # remove the tmp database
    print("deleting tmp database", tmp_name)
    fo.delete_dataset(tmp_name)

    print("\nHAVE A NICE DAY!\n")
    """load with:
    with open(p.output,"r") as f:
        res=json.load(f)
    """
