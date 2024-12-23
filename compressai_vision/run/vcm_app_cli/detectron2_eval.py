# Copyright (c) 2022-2024 InterDigital Communications, Inc
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

from .tools import (
    checkDataset,
    checkForField,
    checkSlice,
    checkVideoDataset,
    checkZoo,
    getQPars,
    loadEncoderDecoderFromPath,
    makeEvalPars,
    setupDetectron2,
    setupVTM,
)


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "detectron2-eval",
        parents=parents,
        help="evaluate model with detectron2 using OpenImageV6",
    )
    required_group = subparser.add_argument_group("required arguments")
    compressai_group = subparser.add_argument_group("compressai-zoo arguments")
    vtm_group = subparser.add_argument_group("vtm arguments")
    optional_group = subparser.add_argument_group("optional arguments")
    required_group.add_argument(
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
    required_group.add_argument(
        "--model",
        action="store",
        type=str,
        required=True,
        default=None,
        nargs="+",
        help="name of Detectron2 config model. It can also be possible to list multiple models.",
    )
    optional_group.add_argument(
        "--output",
        action="store",
        type=str,
        required=False,
        default="compressai-vision.json",
        help="outputfile. Default: compressai-vision.json",
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
    compressai_group.add_argument(
        "--compressai-model-name",
        action="store",
        type=str,
        required=False,
        default=None,
        help="name of an existing model in compressai-zoo. Example: 'cheng2020-attn' ",
    )
    compressai_group.add_argument(
        "--compression-model-path",
        action="store",
        type=str,
        required=False,
        default=None,
        help="a path to a directory containing model.py for custom development model",
    )
    compressai_group.add_argument(
        "--half",
        action="store_true",
        required=False,
        default=False,
        help="convert model to half floating point (fp16)",
    )
    vtm_group.add_argument(
        "--vtm",
        action="store_true",
        default=False,
        help="To enable vtm codec. default: False",
    )
    vtm_group.add_argument(
        "--vtm_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="path to directory with executables EncoderAppStatic & DecoderAppStatic",
    )
    vtm_group.add_argument(
        "--vtm_cfg",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm config file. Example: 'encoder_intra_vtm.cfg' ",
    )
    vtm_group.add_argument(
        "--vtm_cache",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory to cache vtm bitstreams",
    )
    optional_group.add_argument(
        "--qpars",
        action="store",
        type=str,
        required=False,
        default=None,
        help="quality parameters for compressai model or vtm. For compressai-zoo model, it should be integer 1-8. For VTM, it should be integer from 0-51.",
    )
    optional_group.add_argument(
        "--scale",
        action="store",
        type=int,
        required=False,
        default=100,
        help="image scaling as per VCM working group docs. Default: 100",
    )

    optional_group.add_argument(
        "--ffmpeg",
        action="store",
        type=str,
        required=False,
        default="ffmpeg",
        help="path of ffmpeg executable. Default: ffmpeg",
    )
    optional_group.add_argument(
        "--slice",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use a dataset slice instead of the complete dataset. Example: 0:2 for the first two images.  Instead of python slicing string, this can also be a list of sample filepaths in the dataset",
    )
    # subparser.add_argument("--debug", action="store_true", default=False) # not here
    optional_group.add_argument(
        "--progressbar",
        action="store_true",
        default=False,
        help="show fancy progressbar. Default: False",
    )
    optional_group.add_argument(
        "--progress",
        action="store",
        type=int,
        required=False,
        default=1,
        help="Print progress this often",
    )
    optional_group.add_argument(
        "--eval-method",
        action="store",
        type=str,
        required=False,
        default="open-images",
        help="Evaluation method/protocol: open-images or coco.  Default: open-images",
    )
    optional_group.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="Keep tmp databased saved or not.  Default: False",
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

    # dataset.clone needs this
    from compressai_vision import patch  # noqa: F401

    print("fiftyone imported")

    # compressai_vision
    from compressai_vision.evaluation.fo import (  # annex predictions from
        annexPredictions,
        annexVideoPredictions,
    )
    from compressai_vision.evaluation.pipeline import (
        CompressAIEncoderDecoder,
        VTMEncoderDecoder,
    )
    from compressai_vision.pipelines.fo_vcm.constant import vf_per_scale

    # from compressai_vision.pipelines.fo_vcm.tools import getDataFile

    try:
        dataset = fo.load_dataset(p.dataset_name)
    except ValueError:
        print("FATAL: no such registered dataset", p.dataset_name)
        return

    dataset, fr, to = checkSlice(p, dataset)

    compression = True
    # print(">", p.compressai_model_name, p.vtm, p.compression_model_path, p.qpars)
    if (
        (p.compressai_model_name is None)
        and (p.vtm is False)
        and (p.compression_model_path is None)
    ):
        compression = False
        # no (de)compression, just eval
        assert (
            p.qpars is None
        ), "you have provided quality pars but not a (de)compress or vtm model"
        qpars = None  # this indicates no qpars/pure eval run downstream

    else:
        # check quality parameter list
        assert p.qpars is not None, "need to provide integer quality parameters"
        qpars = getQPars(p)

    if p.compressai_model_name is not None:  # compression from compressai zoo
        compression_model = checkZoo(p)
        if compression_model is None:
            print("Can't find compression model")
            return 2

    elif p.compression_model_path is not None:
        encoder_decoder_func = loadEncoderDecoderFromPath(p.compression_model_path)

    elif p.vtm:  # setup VTM
        vtm_encoder_app, vtm_decoder_app, vtm_cfg = setupVTM(p)

    # *** CHOOSE COMPRESSION SCHEME OK ***

    if p.scale is not None:
        assert p.scale in vf_per_scale.keys(), "invalid scale value"

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if p.no_cuda:
        device = "cpu"

    model_names = p.model
    predictors, models_meta, pred_fields = setupDetectron2(model_names, device)

    # instead, create a unique identifier for the field
    # in this run: this way parallel runs dont overwrite
    # each other's field
    # as the database is the same for each running instance/process
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

    eval_method = p.eval_method
    eval_methods = ["open-images", "coco"]
    # must be checked at this stage so that the whole run doesn't crash in the end
    # just because user has fat-fingered the evaluation method
    assert eval_method in eval_methods, "ERROR: allowed eval methods:" + str(
        eval_methods
    )

    if dataset.media_type == "image":
        detection_fields = checkDataset(dataset, fo.core.labels.Detections)
        annex_function = annexPredictions
    elif dataset.media_type == "video":
        detection_fields = checkVideoDataset(dataset, fo.core.labels.Detections)
        annex_function = annexVideoPredictions
    else:
        print("unknown media type", dataset.media_type)
        return

    print()
    print("Using dataset          :", p.dataset_name)

    if p.gt_field not in detection_fields:
        print("FATAL: you have requested field '%s' for ground truths" % (p.gt_field))
        print("     : but it doesn't appear in the dataset samples")
        print("     : please use compressai-vision show to peek at the fields")
        print()
        return 2

    print("Dataset media type     :", dataset.media_type)
    print("Dataset tmp clone      :", tmp_name)
    print("Keep tmp dataset?      :", p.keep)
    print("Image scaling          :", p.scale)
    if p.slice is not None:  # can't use slicing
        print("WARNING: Using slice   :", str(fr) + ":" + str(to))
    print("Number of samples      :", len(dataset))
    print("Torch device           :", device)

    assert len(model_names) == len(models_meta)
    for e, model_info in enumerate(zip(model_names, models_meta)):
        name, meta = model_info
        print(f"=== Vision Model #{e} ====")
        print("Detectron2 model       :", name)
        print("Model was trained with :", meta[0])
        print("Eval. results will be saved to datafield")
        print("                       :", pred_fields[e])
        print("Evaluation protocol    :", eval_method)

    if dataset.media_type == "image":
        classes = dataset.distinct("%s.detections.label" % (p.gt_field))
    elif dataset.media_type == "video":
        classes = dataset.distinct("frames.%s.detections.label" % (p.gt_field))
    classes.sort()
    detectron_classes = copy.deepcopy(meta[1].thing_classes)
    detectron_classes.sort()
    print("Peek model classes     :")
    print(detectron_classes[0:5], "...")
    print("Peek dataset classes   :")
    print(classes[0:5], "...")

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
    """
    if p.compression_model_checkpoint:
        print("WARN: using checkpoint files")
    """
    if qpars is not None:
        print("Quality parameters     :", qpars)
    print("Ground truth data field name")
    print("                       :", p.gt_field)
    if len(detection_fields) > 1:
        print("--> WARNING: you have more than one detection field in your dataset:")
        print(",".join(detection_fields))
        print("be sure to choose the correct one (i.e. for detection or segmentation)")

    # dataset_ = fo.load_dataset(p.dataset_name) # done up there!

    if not checkForField(dataset, p.gt_field):
        return

    # print("(if aborted, start again with --resume=%s)" % predictor_field)
    print("Progressbar            :", p.progressbar)
    if p.progressbar and p.progress > 0:
        print("WARNING: progressbar enabled --> disabling normal progress print")
        p.progress = 0
    print("Print progress         :", p.progress)
    print("Output file            :", p.output)

    if not p.y:
        input("press enter to continue.. ")

    # save metadata about the run into the json file
    metadata = {
        "dataset": p.dataset_name,
        "gt_field": p.gt_field,
        "tmp-dataset": tmp_name,
        "slice": p.slice,
        "model": model_names,
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

    """
    # parameters for dataset.evaluate_detections
    # https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#evaluating-videos
    eval_args = {"gt_field": p.gt_field, "method": eval_method}
    if eval_method == "open-images":
        if dataset.get_field("positive_labels"):
            eval_args["pos_label_field"] = "positive_labels"
        if dataset.get_field("negative_labels"):
            eval_args["neg_label_field"] = "negative_labels"
        eval_args["expand_pred_hierarchy"] = False
        eval_args["expand_gt_hierarchy"] = False
    else:
        eval_args["compute_mAP"] = True
    """
    pred_fields_, eval_args = makeEvalPars(
        dataset=dataset,
        gt_field=p.gt_field,
        predictor_fields=pred_fields,
        eval_method=eval_method,
    )
    # print(predictor_field_)
    # print(eval_args)

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

    if compression:
        for quality in qpars:
            enc_dec = None  # default: no encoding/decoding
            if (
                p.compressai_model_name or p.compression_model_path
            ):  # compressai model, either from the zoo or from a directory:
                if p.compressai_model_name is not None:
                    # e.g. "bmshj2018-factorized"
                    print("\nQUALITY PARAMETER: ", quality)
                    net = (
                        compression_model(quality=quality, pretrained=True)
                        .eval()
                        .to(device)
                    )
                    if p.half:
                        net = net.half()
                    enc_dec = CompressAIEncoderDecoder(
                        net,
                        device=device,
                        scale=p.scale,
                        ffmpeg=p.ffmpeg,
                        dump=p.dump,
                        half=p.half,
                    )
                else:  # or a custom model from a file:
                    enc_dec = encoder_decoder_func(
                        quality=quality,
                        device=device,
                        scale=p.scale,
                        ffmpeg=p.ffmpeg,
                        dump=p.dump,
                        half=p.half,
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
            enc_dec.computeMetrics(False)

            # append pred_fields_ with the quality point tag
            # i.e. detectron-eval_v0 ==> detectron-eval_v0-qp-1
            # this way, if we use the --keep flag, we can
            # see detection results for different qpoints
            # in the fo app
            pred_fields_qp = []
            for pred_field_ in pred_fields_:
                pred_fields_qp.append(pred_field_ + "-qp-" + str(quality))

            bpp = annex_function(
                predictors=predictors,
                fo_dataset=dataset,
                encoder_decoder=enc_dec,
                gt_field=p.gt_field,
                predictor_fields=pred_fields_qp,
                use_pb=p.progressbar,
                use_print=p.progress,
            )

            if bpp is None or bpp < 0:
                print()
                print("Sorry, mAP calculation aborted")
                # TODO: implement:
                # print("If you want to resume, start again with:")
                # print("--continue", predictor_field)
                print()
                return

            if not p.progressbar:
                fo.config.show_progress_bars = False

            bpps = []
            accs = []
            accs_detail = []
            for _, pred_field_ in enumerate(pred_fields_qp):
                # print("evaluating dataset", dataset.name)
                res = dataset.evaluate_detections(pred_field_, **eval_args)
                bpps.append(bpp)
                accs.append(res.mAP())
                accs_detail.append(per_class(res))

            xs.append(bpps)
            ys.append(accs)
            maps.append(accs_detail)

    else:  # a pure evaluation without encoding/decoding
        bpp = annex_function(
            predictors=predictors,
            fo_dataset=dataset,
            gt_field=p.gt_field,
            predictor_fields=pred_fields_,
            use_pb=p.progressbar,
            use_print=p.progress,
        )

        bpps = []
        accs = []
        accs_detail = []
        for _, pred_field_ in enumerate(pred_fields_):
            # print("evaluating dataset", dataset.name)
            res = dataset.evaluate_detections(
                pred_field_,
                **eval_args,
                # gt_field=p.gt_field,
                # method="open-images",
                # pos_label_field="positive_labels",
                # neg_label_field="negative_labels",
                # expand_pred_hierarchy=False,
                # expand_gt_hierarchy=False,
            )
            bpps.append(bpp)
            accs.append(res.mAP())
            accs_detail.append(per_class(res))

        xs.append(bpps)
        ys.append(accs)
        maps.append(accs_detail)

    # print(">>", metadata)
    metadata["bpp"] = xs
    metadata["map"] = ys
    metadata["map_per_class"] = maps
    with open(p.output, "w") as f:
        f.write(json.dumps(metadata, indent=2))

    """maybe not?
    print("\nResult output:")
    print(json.dumps(metadata, indent=2))
    """
    if not p.keep:
        # remove the tmp database
        print("deleting tmp database", tmp_name)
        fo.delete_dataset(tmp_name)
    else:
        print("keeping tmp database", tmp_name)

    print("\nDone!\n")
    """load with:
    with open(p.output,"r") as f:
        res=json.load(f)
    """
