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


def main(p):  # noqa: C901
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

    assert p.dataset_name is not None, "please provide dataset name"
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

    assert p.model is not None, "provide Detectron2 model name"

    if (p.compressai is None) == p.vtm == (p.modelpath is None) == False:
        compression = False
        # no (de)compression, just eval
        assert (
            qpars is None
        ), "you have provided quality pars but not a (de)compress model"

    # check that only one is defined
    defined = False
    for scheme in [p.compressai, p.vtm, p.modelpath]:
        if scheme:
            if defined:  # second match!
                raise AssertionError(
                    "please define only one of the following: compressai, vtm or modelpath"
                )
            defined = True

    if defined:
        # check quality parameter list
        assert p.qpars is not None, "need to provide integer quality parameters"
        try:
            qpars = [int(i) for i in p.qpars.split(",")]
        except Exception as e:
            print("problems with your quality parameter list")
            raise e
        # check checkpoint file validity if defined
        if p.checkpoint is not None:
            assert os.path.isfile(p.checkpoint), "can't find defined checkpoint file"

    else:
        qpars = None

    # *** CHOOSE COMPRESSION SCHEME ***

    if p.compressai is not None:  # compression from compressai zoo
        import compressai.zoo

        # compressai_model = getattr(compressai.zoo, "bmshj2018_factorized")
        compressai_model = getattr(
            compressai.zoo, p.compressai
        )  # a function that returns a model instance or just a class

    elif p.modelpath is not None:  # compression from a custcom compressai model
        path = os.path.join(p.modelpath, "model.py")
        assert os.path.isfile(path), "your model directory is missing model.py"
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location(
                "module", os.path.join(p.modelpath, path)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print("loading model from directory", p.modelpath, "failed with", e)
            return
        else:
            assert hasattr(
                module, "getModel"
            ), "your module is missing getModel function"
            compressai_model = (
                module.getModel
            )  # a function that returns a model instance or just a class

    elif p.vtm is not None:  # setup VTM
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
    if p.compressai is not None:
        print("Using compressai model :", p.compressai)
    elif p.modelpath is not None:
        print("Using custom mode from :", p.modelpath)
    elif p.vtm:
        print("Using VTM               ")
        if p.vtm_cache:
            # assert(os.path.isdir(p.vtm_cache)), "no such directory "+p.vtm_cache
            # ..created by the VTMEncoderDecoder class
            print("WARNING: VTM USES CACHE IN", p.vtm_cache)
    else:
        print("** Evaluation without Encoding/Decoding **")
    if p.checkpoint:
        print("WARN: using checkpoint :", p.checkpoint)
    if qpars is not None:
        print("Quality parameters     :", qpars)
    print("Eval. datafield name   :", predictor_field)
    # print("(if aborted, start again with --resume=%s)" % predictor_field)
    print("Progressbar            :", p.progressbar)
    if p.progressbar and p.progress > 0:
        print("WARNING: progressbar enabled --> disabling normal progress print")
        p.progress = 0
    print("Print progress         :", p.progress)
    print("Output file            :", p.output)
    classes = dataset.distinct("detections.detections.label")
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
        "tmp datasetname": tmp_name,
        "slice": p.slice,
        "model": model_name,
        "compressai model": p.compressai,
        "custom model": p.modelpath,
        "checkpoint": p.checkpoint,
        "vtm": p.vtm,
        "vtm_cache": p.vtm_cache,
        "qpars": qpars,
    }
    with open(p.output, "w") as f:
        json.dump(metadata, f)

    # please see ../monkey.py for problems I encountered when cloning datasets
    # simultaneously with various multiprocesses/batch jobs
    print("cloning dataset", p.dataset_name, "to", tmp_name)
    dataset = dataset.clone(tmp_name)
    dataset.persistent = True
    # fo.core.odm.database.sync_database() # this would've helped? not sure..

    print("instantiating Detectron2 predictor")
    predictor = DefaultPredictor(cfg)

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
    # bpp, mAP values, mAP breakdown per class
    if qpars is not None:
        # loglev=logging.DEBUG # this now set in main
        # loglev = logging.INFO
        # quickLog("CompressAIEncoderDecoder", loglev)
        # quickLog("VTMEncoderDecoder", loglev)
        for i in qpars:
            print("\nQUALITY PARAMETER", i)
            enc_dec = None  # default: no encoding/decoding
            if (
                p.compressai or p.modelpath
            ):  # compressai model, either from the zoo or from a directory
                if p.checkpoint is None:
                    net = compressai_model(quality=i, pretrained=True).eval().to(device)
                    # e.g. compressai.zoo.bmshj2018_factorized
                    # or a custom model form a file
                else:  # load a checkpoint
                    net = compressai_model(quality=i)
                    try:
                        cp = torch.load(p.checkpoint)
                        # print(">>>", cp.keys())
                        # net.load_state_dict(cp['state_dict']).eval().to(device)
                        net.load_state_dict(cp).eval().to(device)
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
                    qp=i,
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
                gt_field="detections",
                method="open-images",
                pos_label_field="positive_labels",
                neg_label_field="negative_labels",
                expand_pred_hierarchy=False,
                expand_gt_hierarchy=False,
            )
            xs.append(bpp)
            ys.append(res.mAP())
            maps.append(per_class(res))

    else:  # a pure evaluation without encoding/decoding
        bpp = annexPredictions(
            predictor=predictor,
            fo_dataset=dataset,
            predictor_field=predictor_field,
            use_pb=p.progressbar,
            use_print=p.progress,
        )
        res = dataset.evaluate_detections(
            predictor_field,
            gt_field="detections",
            method="open-images",
            pos_label_field="positive_labels",
            neg_label_field="negative_labels",
            expand_pred_hierarchy=False,
            expand_gt_hierarchy=False,
        )
        xs.append(bpp)
        ys.append(res.mAP())
        maps.append(per_class(res))

    # print(">>", metadata)
    metadata["bpp"] = xs
    metadata["map"] = ys
    metadata["map_per_class"] = maps
    with open(p.output, "w") as f:
        json.dump(metadata, f)

    # remove the tmp database
    print("deleting tmp database", tmp_name)
    fo.delete_dataset(tmp_name)

    print("\nHAVE A NICE DAY!\n")
    """load with:
    with open(p.output,"r") as f:
        res=json.load(f)
    """
