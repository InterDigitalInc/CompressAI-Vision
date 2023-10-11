 
def getCfgPredictor():
    """Returns a tuple with:

    cfg: detectron2.config, predictor: detectron2.DefaultPredictor
    """
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    # from detectron2.data import MetadataCatalog  # , DatasetCatalog

    # TODO: set your particular paths
    model_pth="/home/sampsa/silo/interdigital/sync/MPEG_FLIR_anchor/dataset/fine_tuned_model/model_final.pth"
    coco_format_json_filename = "/home/sampsa/silo/interdigital/sync/MPEG_FLIR_anchor/dataset/coco_format_json_annotation/FLIR_val_thermal_coco_format_jpg.json"
    """
    try:
        register_coco_instances("FLIR_val", {}, coco_format_json_filename, "/dummy") # path to images doesn't really matter, we just need the metadata
    except AssertionError:
        print("FLIR_val ALREADY REG")
    """
    register_coco_instances("FLIR_val", {}, coco_format_json_filename, "/dummy") # path to images doesn't really matter, we just need the metadata

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_pth
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TRAIN = ("FLIR_val",)  # This is required to set the metadata format from the Trainer.
    
    # some Detectron2 bs:
    # we need to instantiate a trainer so that cfg.thing_classes becomes populated
    # but the I came across this one:
    # AttributeError: module 'distutils' has no attribute 'version'
    # see: https://github.com/facebookresearch/detectron2/issues/3811
    try:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
    except AttributeError:
        # doesn't really matter since we're not interested in the trainer anyway
        pass

    predictor = DefaultPredictor(cfg)
    # meta=MetadataCatalog.get("FLIR_val")
    # print("2>>>", meta.thing_classes)
    return cfg, predictor
