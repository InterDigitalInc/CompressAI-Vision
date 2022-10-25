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
    if p.slice is not None:
        print(
            "WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?"
        )
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
    return dataset, fr, to


def setupDetectron2(model_name, device):
    # *** Detectron imports ***
    # Some basic setup:
    # Setup detectron2 logger
    # import detectron2
    from detectron2.utils.logger import setup_logger

    setup_logger()

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog  # , DatasetCatalog

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

    return cfg, model_meta, model_dataset


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
