
import cv2
import math, os
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
        encoder_decoder = None # compressai_vision.evaluation.pipeline.base.EncoderDecoder
        ):
    """Run detector on a dataset and annex predictions of the detector into the dataset.

    :param predictor: A Detectron2 predictor
    :param fo_dataset: Fiftyone dataset
    :param detection_field: Which dataset member to use for ground truths.  Default: "detections"
    :param predictor_field: Which dataset member to use for saving the Detectron2 results.  Default: "detectron-predictions"
    :param encoder_decoder: (optional) a ``compressai_vision.evaluation.pipeline.EncoderDecoder`` subclass instance to apply on the image before detection
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
            path = sample.filepath
            im = cv2.imread(path)
            tag=path.split(os.path.sep)[-1].split(".")[0] # i.e.: /path/to/some.jpg --> some.jpg --> some
            # print(tag)
            if encoder_decoder is not None:
                # before using detector, crunch through 
                # encoder/decoder
                bpp, im=encoder_decoder.BGR(im, tag=tag) # include a tag for cases where EncoderDecoder uses caching
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
