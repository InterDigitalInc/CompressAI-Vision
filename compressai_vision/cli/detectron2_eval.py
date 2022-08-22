"""cli detectron2_eval functionality
""" 
import logging, pickle, copy, json
# fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
# compressai_vision
from compressai_vision.evaluation.fo import annexPredictions # annex predictions from
from compressai_vision.evaluation.pipeline import CompressAIEncoderDecoder, VTMEncoderDecoder
from compressai_vision.tools import quickLog


def main(p):
    assert(p.name is not None), "please provide dataset name"
    try:
        dataset=fo.load_dataset(p.name)
    except ValueError:
        print("FATAL: no such registered database", p.name)
        return
    assert(p.model is not None), "provide Detectron2 model name"    

    qpars = None
    if (p.compressai is not None):
        if p.vtm:
            print("FATAL: evaluation either with compressai or vtm or with nothing")
            return
        assert(p.qpars is not None), "need to provide quality parameters for compressai"
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

    if (p.vtm):
        assert(p.qpars is not None), "need to provide quality parameters for vtm"
        try:
            qpars = [float(i) for i in p.qpars.split(",")]
        except Exception as e:
            print("problems with your quality parameter list")
            raise e
    
    if ((p.vtm is None) and (p.compressai is None)) and (p.qpars is not None):
        print("FATAL: you defined qpars although they are not needed")
        return

    # compressai_model == None --> no compressai
    # p.vtm == False --> no vtm

    ## *** Detectron imports ***
    # Some basic setup:
    # Setup detectron2 logger
    import detectron2
    import torch
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    model_name = p.model

    # cfg encapsulates the model architecture & weights, also threshold parameter, metadata, etc.
    cfg = get_cfg()
    cfg.MODEL.DEVICE=device
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
    model_dataset=cfg.DATASETS.TRAIN[0]
    # print("model was trained with", model_dataset)
    model_meta=MetadataCatalog.get(model_dataset)

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

    classes = dataset.distinct(
        "detections.detections.label"
    )
    classes.sort()
    detectron_classes=copy.deepcopy(model_meta.thing_classes)
    detectron_classes.sort()
    print("Peek model classes     :")
    print(detectron_classes[0:5], "...")
    print("Peek dataset classes   :")
    print(classes[0:5], "...")

    if not p.y:
        input('press enter to continue.. ')

    print("instantiating Detectron2 predictor")
    predictor = DefaultPredictor(cfg)

    predictor_field="detectron-predictions"
    
    def per_class(results_obj):
        """take fiftyone/openimagev6 results object & spit
        out mAP breakdown as per class
        """
        d = {}
        for class_ in classes:
            d[class_] = results_obj.mAP([class_])
        return d

    xs=[]; ys=[]; maps=[]; # bpp, mAP values, mAP breakdown per class

    if qpars is not None:
        # loglev=logging.DEBUG
        loglev=logging.INFO
        quickLog("CompressAIEncoderDecoder", loglev)
        quickLog("VTMEncoderDecoder", loglev)
        for i in qpars: 
            # concurrency considerations
            # could multithread/process over quality pars
            # .. but that would be even better to do at the cli level
            # beware that all processes use the same fiftyone/mongodb, so maybe
            # different predictor_field instance for each running (multi)process
            print("\nQUALITY PARAMETER", i)
            if compressai_model is not None:
                net = compressai_model(quality=i, pretrained=True).eval().to(device)
                enc_dec = CompressAIEncoderDecoder(net, device=device)
            else:
                raise AssertionError("JACKY-TODO")
                enc_dec = VTMEncoderDecoder()
            bpp = annexPredictions(
                predictor=predictor, 
                fo_dataset=dataset, 
                encoder_decoder=enc_dec, 
                predictor_field=predictor_field)
            res = dataset.evaluate_detections(
                predictor_field,
                gt_field="detections",
                method="open-images",
                pos_label_field="positive_labels",
                neg_label_field="negative_labels",
                expand_pred_hierarchy=False,
                expand_gt_hierarchy=False
            )
            xs.append(bpp)
            ys.append(res.mAP())
            maps.append(per_class(res))
            with open(p.output,"w") as f:
                json.dump({
                    "bpp" : xs, 
                    "map" : ys,
                    "map_per_class" : maps
                    }, f)

    else:
        bpp = annexPredictions(
            predictor=predictor, 
            fo_dataset=dataset,
            predictor_field=predictor_field)
        res = dataset.evaluate_detections(
            predictor_field,
            gt_field="detections",
            method="open-images",
            pos_label_field="positive_labels",
            neg_label_field="negative_labels",
            expand_pred_hierarchy=False,
            expand_gt_hierarchy=False
        )
        xs.append(bpp)
        ys.append(res.mAP())
        maps.append(per_class(res))
        """ # let's use json instead
        with open(p.output,"wb") as f:
            pickle.dump((xs, ys, maps), f)
        """
        with open(p.output,"w") as f:
            json.dump({
                "bpp" : xs, 
                "map" : ys,
                "map_per_class" : maps
                }, f)

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
