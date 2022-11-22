import os

from pathlib import Path


def getQPars(p):
    try:
        qpars = [int(i) for i in p.qpars.split(",")]
    except Exception as e:
        print("problems with your quality parameter list")
        raise e
    return qpars


def loadEncoderDecoderFromPath(compression_model_path):
    # compression from a custcom compression model
    model_file = Path(compression_model_path) / "model.py"
    if model_file.is_file():
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location("module", model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(
                "loading model from directory",
                compression_model_path,
                "failed with",
                e,
            )
            raise
        else:
            assert hasattr(
                module, "getEncoderDecoder"
            ), "your module is missing getEncoderDecoder function"
            encoder_decoder_func = (
                module.getEncoderDecoder
            )  # a function that returns EncoderDecoder instance
            print("loaded custom model.py")
    else:
        raise FileNotFoundError(f"No model.py in {compression_model_path}")
    return encoder_decoder_func


def setupVTM(p):
    if p.vtm_dir is None:
        try:
            vtm_dir = os.environ["VTM_DIR"]
        except KeyError as e:
            print("please define --vtm_dir or set environmental variable VTM_DIR")
            raise e
    else:
        vtm_dir = p.vtm_dir

    vtm_dir = os.path.expanduser(vtm_dir)

    if p.vtm_cfg is None:
        # vtm_cfg = getDataFile("encoder_intra_vtm_1.cfg")
        # print("WARNING: using VTM default config file", vtm_cfg)
        raise BaseException("VTM config is not defined")
    else:
        vtm_cfg = p.vtm_cfg
    vtm_cfg = os.path.expanduser(vtm_cfg)  # some more systematic way of doing these..
    print("Reading vtm config from: " + vtm_cfg)
    assert os.path.isfile(vtm_cfg), "vtm config file not found"
    # try both filenames..
    vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStatic")
    if not os.path.isfile(vtm_encoder_app):
        vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStaticd")
    if not os.path.isfile(vtm_encoder_app):
        raise AssertionError("FATAL: can't find EncoderAppStatic(d) in " + vtm_dir)
    # try both filenames..
    vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStatic")
    if not os.path.isfile(vtm_decoder_app):
        vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStaticd")
    if not os.path.isfile(vtm_decoder_app):
        raise AssertionError("FATAL: can't find DecoderAppStatic(d) in " + vtm_dir)

    return vtm_encoder_app, vtm_decoder_app, vtm_cfg


def checkSlice(p, dataset):
    fr = None
    to = None
    if p.slice is not None:
        print(
            "WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?"
        )
        # say, 0:100
        nums = p.slice.split(":")  # python slicing?
        """
        if len(nums) < 2:
            print("invalid slicing: use normal python slicing, say, 0:100")
            return
        """
        filepaths = p.slice.split(",")  # list of filepaths?
        if len(nums) >= 2:
            # looks like 0:N slice
            try:
                fr = int(nums[0])
                to = int(nums[1])
            except ValueError:
                print("invalid slicing: use normal python slicing, say, 0:100")
                raise
            assert to > fr, "invalid slicing: use normal python slicing, say, 0:100"
            dataset = dataset[fr:to]
        elif len(filepaths) >= 1:
            from fiftyone import ViewField as F

            query_list = []
            # looks like list of filenames
            for filepath in filepaths:
                p = Path(filepath).expanduser().absolute()
                if not p.is_file():
                    print("FATAL: file", str(p), "does not exist")
                    raise AttributeError("file not found or --slice misinterpretation")
                query_list.append(F("filepath") == str(p))
            dataset = dataset[F.any(query_list)]
        else:
            print("could not interprete --slice")
            raise AttributeError("could not interprete --slice")
    return dataset, fr, to


def setupDetectron2(model_names: list, device):
    # Parsing a list of detectron2 models names and return instantiated model with meta information
    # *** Detectron imports ***
    # Some basic setup:
    # Setup detectron2 logger
    # import detectron2
    import logging
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    logger.setLevel(logging.WARNING)

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog  # , DatasetCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.data.datasets import register_coco_instances

    models = []
    models_meta = []
    pred_fields = []
    for e, name in enumerate(model_names):
        if ".py" in name:
            # *** LOAD cfg and predictor from an external .py file ***
            pyfile = name
            print("Trying to load custom Detectron2 model from local file", pyfile)
            assert os.path.exists(pyfile), "Can't find " + str(pyfile)
            import importlib.util

            try:
                spec = importlib.util.spec_from_file_location(
                    "detectron2_module", pyfile
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                print("Importing custom detectron model from", pyfile, "failed")
                raise
            assert hasattr(module, "getCfgPredictor"), (
                "file " + pyfile + " is missing function getCfgPredictor"
            )
            print("Loading custom Detectron2 predictor from", pyfile)
            """
            cfg, predictor = module.getCfgPredictor(
                get_cfg,
                model_zoo,
                register_coco_instances,
                DefaultPredictor,
                MetadataCatalog
            )
            """
            cfg, predictor = module.getCfgPredictor()
            model_dataset = cfg.DATASETS.TRAIN[0]
            model_meta = MetadataCatalog.get(model_dataset)
        else:
            # *** LOAD cfg and predictor from the Detectron2 zoo ***
            # cfg encapsulates the model architecture & weights, also threshold parameter, metadata, etc.
            cfg = get_cfg()
            cfg.MODEL.DEVICE = device
            # load config from a file:
            cfg.merge_from_file(model_zoo.get_config_file(name))
            # DO NOT TOUCH THRESHOLD WHEN DOING EVALUATION:
            # too big a threshold will cut the smallest values
            # & affect the precision(recall) curves & evaluation results
            # the default value is 0.05
            # value of 0.01 saturates the results (they don't change at lower values)
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            # get weights
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
            # print("expected input colorspace:", cfg.INPUT.FORMAT)
            # print("loaded datasets:", cfg.DATASETS)
            model_dataset = cfg.DATASETS.TRAIN[0]
            # print("model was trained with", model_dataset)
            model_meta = MetadataCatalog.get(model_dataset)
            print(f"instantiating Detectron2 predictor {e} : {name}")
            predictor = DefaultPredictor(cfg)

        models.append(predictor)
        models_meta.append((model_dataset, model_meta))
        pred_fields.append(f"detectron-predictions_v{e}")

    return models, models_meta, pred_fields


def checkDataset(dataset, doctype):
    """Look for fields of certain type in the dataset, say, of type
    fiftyone.core.labels.Detections.

    Return a list of matching field names
    """
    keys = []
    for key in dataset.get_field_schema():
        df = dataset.get_field(key)
        if hasattr(df, "document_type") and df.document_type == doctype:
            # print(df)
            # df.document_type == fiftyone.core.labels.Detections
            keys.append(key)
    return keys


def checkVideoDataset(dataset, doctype):
    """Look for fields of certain type in the dataset, say, of type
    fiftyone.core.labels.Detections.

    Return a list of matching field names
    """
    keys = []
    for key in dataset.get_frame_field_schema():
        df = dataset.get_field(key)
        if hasattr(df, "document_type") and df.document_type == doctype:
            # print(df)
            # df.document_type == fiftyone.core.labels.Detections
            keys.append(key)
    return keys


def checkZoo(p):
    from compressai.zoo import models

    try:
        compression_model = models[p.compressai_model_name]
    except KeyError:
        print(f"Supported model names are {models.keys()}")
        return
    return compression_model


def checkForField(dataset, name):
    if dataset.media_type == "image":
        if dataset.get_field(name) is None:
            print("FATAL: your dataset does not have requested field '" + name + "'")
            print("Dataset info:")
            print(dataset)
            return False
    elif dataset.media_type == "video":
        if name in dataset.get_frame_field_schema():
            pass
        else:
            print(
                "FATAL: your video dataset's frames do not not have requested field '"
                + name
                + "'"
            )
            print("Dataset info:")
            print(dataset)
            return False
    else:
        print("FATAL: unknow media type", dataset.media_type)
        return False
    return True


def makeEvalPars(dataset=None, gt_field=None, predictor_fields=None, eval_method=None):
    """Make parameters for Dataset.evaluate_detections method

    Refs:

    - https://voxel51.com/docs/fiftyone/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.evaluate_detections
    - https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#evaluating-videos

    For images & with open image protocol:

    ::

        dataset.evaluate_detections(
            "predictions",
            gt_field=p.gt_field,
            method="open-images",
            pos_label_field="positive_labels",
            neg_label_field="negative_labels",
            expand_pred_hierarchy=False,
            expand_gt_hierarchy=False
            )


    For videos: note the extra "frames.":

    ::

        dataset.evaluate_detections(
            "frames.predictions",
            gt_field="frames.detections",
            eval_key="eval"
            )

    returns args: str, kwargs: dict
    """
    if dataset.media_type == "image":
        pred_fields_ = predictor_fields
        eval_args = {"gt_field": gt_field, "method": eval_method}
        if eval_method == "open-images":
            if dataset.get_field("positive_labels"):
                eval_args["pos_label_field"] = "positive_labels"
            if dataset.get_field("negative_labels"):
                eval_args["neg_label_field"] = "negative_labels"
            eval_args["expand_pred_hierarchy"] = False
            eval_args["expand_gt_hierarchy"] = False
        else:
            eval_args["compute_mAP"] = True

    elif dataset.media_type == "video":
        pred_fields_ = ["frames." + field for field in predictor_fields]
        eval_args = {"gt_field": "frames." + gt_field, "method": eval_method}
        if eval_method == "open-images":
            if "positive_labels" in dataset.get_frame_field_schema():
                eval_args["pos_label_field"] = "positive_labels"
            if "negative_label" in dataset.get_frame_field_schema():
                eval_args["neg_label_field"] = "negative_labels"
            eval_args["expand_pred_hierarchy"] = False
            eval_args["expand_gt_hierarchy"] = False
        else:
            eval_args["compute_mAP"] = True

    return pred_fields_, eval_args


def makeVideoThumbnails(dataset, force=False):
    """
    :param dataset: video dataset
    """
    import fiftyone.utils.video as fouv

    for sample in dataset.iter_samples(progress=True):
        sample_dir = os.path.dirname(sample.filepath)
        output_path = os.path.join(sample_dir, "web_" + sample.filename)
        if (not force) and os.path.isfile(output_path):
            print("WARNING: file", output_path, "already exists - will skip")
            continue
        print("\nRe-encoding", sample.filepath, "to", output_path)
        fouv.reencode_video(sample.filepath, output_path)
        sample["web_filepath"] = output_path
        sample.save()
