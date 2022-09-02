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
<<<<<<< HEAD
import copy, os, uuid, datetime
=======
import copy
>>>>>>> main
import json
import os

# fiftyone
import fiftyone as fo

# compressai_vision
from compressai_vision.evaluation.fo import annexPredictions  # annex predictions from
from compressai_vision.evaluation.pipeline import (
    CompressAIEncoderDecoder,
    VTMEncoderDecoder,
)
from compressai_vision.tools import getDataFile


def main(p):  # noqa: C901
    assert p.name is not None, "please provide dataset name"
    try:
        dataset = fo.load_dataset(p.name)
    except ValueError:
        print("FATAL: no such registered database", p.name)
        return
    assert p.model is not None, "provide Detectron2 model name"

    qpars = None
    if p.compressai is not None:
        if p.vtm:
            print("FATAL: evaluation either with compressai or vtm or with nothing")
            return
        assert p.qpars is not None, "need to provide quality parameters for compressai"
        try:
            qpars = [int(i) for i in p.qpars.split(",")]
        except Exception as e:
            print("problems with your quality parameter list")
            raise e
        import compressai.zoo

        # compressai_model = getattr(compressai.zoo, "bmshj2018_factorized")
        compressai_model = getattr(compressai.zoo, p.compressai)
    else:
        compressai_model = None

    if p.vtm:
        assert p.qpars is not None, "need to provide quality parameters for vtm"
        try:
            qpars = [float(i) for i in p.qpars.split(",")]
        except Exception as e:
            print("problems with your quality parameter list")
            raise e
        if p.vtm_dir is None:
            try:
                vtm_dir = os.environ["VTM_DIR"]
            except KeyError as e:
                print("please define --vtm_dir or set environmental variable VTM_DIR")
                # raise e
                return
        else:
            vtm_dir = p.vtm_dir

        if p.vtm_cfg is None:
            vtm_cfg = getDataFile("encoder_intra_vtm_1.cfg")
            print("WARNING: using VTM default config file", vtm_cfg)
        else:
            vtm_cfg = p.vtm_cfg
            assert os.path.isfile(vtm_cfg), "vtm config file not found"

        vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStatic")
        vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStatic")

    if ((p.vtm is None) and (p.compressai is None)) and (p.qpars is not None):
        print("FATAL: you defined qpars although they are not needed")
        return

    # compressai_model == None --> no compressai
    # p.vtm == False --> no vtm

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

    print()
    print("Using dataset          :", p.name)
    print("Number of samples      :", len(dataset))
    print("Torch device           :", device)
    print("Detectron2 model       :", model_name)
    print("Model was trained with :", model_dataset)
    if compressai_model is not None:
        print("Using compressai model :", p.compressai)
    elif p.vtm:
        print("Using VTM               ")
    else:
        print("** Evaluation without Encoding/Decoding **")
    if qpars is not None:
        print("Quality parameters      :", qpars)

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

    print("instantiating Detectron2 predictor")
    predictor = DefaultPredictor(cfg)

    # predictor_field = "detectron-predictions"
    ## instead, create a unique identifier for the field
    ## in this run: this way parallel runs dont overwrite
    ## each other's field
    ## as the database is the same for each running instance/process
    # ui=uuid.uuid1().hex # 'e84c73f029ee11ed9d19297752f91acd'
    # predictor_field = "detectron-"+ui
    predictor_field = "detectron-{0:%Y-%m-%d-%H-%M-%S-%f}".format(
        datetime.datetime.now()
    )

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

    dataset.persistent = False

    if qpars is not None:
        # loglev=logging.DEBUG # this now set in main
        # loglev = logging.INFO
        # quickLog("CompressAIEncoderDecoder", loglev)
        # quickLog("VTMEncoderDecoder", loglev)
        for i in qpars:
            # concurrency considerations
            # could multithread/process over quality pars
            # .. but that would be even better to do at the cli level
            # beware that all processes use the same fiftyone/mongodb, so maybe
            # different predictor_field instance for each running (multi)process
            print("\nQUALITY PARAMETER", i)
            enc_dec = None  # default: no encoding/decoding
            if compressai_model is not None:
                net = compressai_model(quality=i, pretrained=True).eval().to(device)
                enc_dec = CompressAIEncoderDecoder(net, device=device)
<<<<<<< HEAD
            # elif p.vtm:
            else:  # eh.. must be VTM
=======
            else:
>>>>>>> main
                enc_dec = VTMEncoderDecoder(
                    encoderApp=vtm_encoder_app,
                    decoderApp=vtm_decoder_app,
                    ffmpeg=p.ffmpeg,
                    vtm_cfg=vtm_cfg,
                    qp=i,
                )
<<<<<<< HEAD
            print("predictor_field=", predictor_field)
=======
>>>>>>> main
            bpp = annexPredictions(
                predictor=predictor,
                fo_dataset=dataset,
                encoder_decoder=enc_dec,
                predictor_field=predictor_field,
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
            with open(p.output, "w") as f:
                json.dump({"bpp": xs, "map": ys, "map_per_class": maps}, f)

    else:
        bpp = annexPredictions(
            predictor=predictor, fo_dataset=dataset, predictor_field=predictor_field
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
        """ # let's use json instead
        with open(p.output,"wb") as f:
            pickle.dump((xs, ys, maps), f)
        """
        with open(p.output, "w") as f:
            json.dump({"bpp": xs, "map": ys, "map_per_class": maps}, f)

    # remove the predicted field from the database
    dataset.delete_sample_field(predictor_field)

    print("\nHAVE A NICE DAY!\n")
    """load with:
    with open(p.output,"r") as f:
        res=json.load(f)
    """
    """old:
    with open(p.output,"rb") as f:
        xs, ys, maps = pickle.load(f)
        print(xs, ys, maps)
    """
