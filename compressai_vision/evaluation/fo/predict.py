
import cv2
import math
from detectron2.data import MetadataCatalog
from compressai_vision.conversion.detectron2 import findLabels, detectron251
from compressai_vision.evaluation.pipeline.base import EncoderDecoder
from fiftyone.core.dataset import Dataset
from fiftyone import ProgressBar


def annexPredictions(
        predictor = None,
        fo_dataset: Dataset = None,
        detection_field: str = "detections",
        predictor_field: str = "detectron-predictions",
        encoder_decoder = None, # compressai_vision.evaluation.pipeline.base.EncoderDecoder
        append = None
        ):
    """Run detector and annex predictions under the key "detectron-predictions" of each sample

    :param predictor: Detector2 predictor
    :param fo_dataset: Fiftyone dataset
    :detection_field: Which member to use for ground truths
    :encoder_decoder: (optional) an EncoderDecoder instance to cruch the image through before
    detection
    :append: don't overwrite the default "detectron-predictions" field, but add a new
    field instead.  Default=None: don't append any new field
    """

    assert(predictor is not None), "provide Detectron2 predictor"
    assert(fo_dataset is not None), "provide fiftyone dataset"
    if encoder_decoder is not None:
        assert(issubclass(encoder_decoder.__class__, EncoderDecoder)),\
            "encoder_decoder instances needs to be a subclass of EncoderDecoder"
    
    model_meta=MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])

    """we don't need this!
    d2_dataset = FO2DetectronDataset(
        fo_dataset=fo_dataset,
        detection_field=detection_field,
        model_catids = model_meta.things_classes,
        )
    """
    try:
        allowed_labels=findLabels(fo_dataset, detection_field=detection_field)
    except ValueError:
        print("your ground truths suck: samples have no member", detection_field, "will set allowed_labels to empty list")
        allowed_labels=[]

    # add predicted bboxes to each fiftyone Sample
    bpp_sum=0
    with ProgressBar(fo_dataset) as pb:
        for sample in fo_dataset:
            # sample.filepath
            im = cv2.imread(sample.filepath)
            if encoder_decoder is not None:
                # before using detector, crunch through 
                # encoder/decoder
                bpp, im=encoder_decoder.BGR(im)
                bpp_sum += bpp
            res = predictor(im)
            predictions=detectron251(res, 
                model_catids=model_meta.thing_classes, 
                # allowed_labels=allowed_labels # not needed, really
            ) # fiftyone Detections object
            sample[predictor_field]=predictions
            ## we could attach the bitrate to each detection, of course
            #if encoder_decoder is not None:
            #    sample["bpp"]=bpp
            sample.save()
            pb.update()
    if encoder_decoder is None:
        return None
    elif len(fo_dataset) > 0:
        return bpp_sum/len(fo_dataset)
    else:
        return math.nan
