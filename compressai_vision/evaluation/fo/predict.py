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

import traceback

import cv2

from detectron2.data import MetadataCatalog
from fiftyone import ProgressBar
from fiftyone.core.dataset import Dataset

from compressai_vision.conversion.detectron2 import (
    detectron251,
    findLabels,
    findVideoLabels,
)
from compressai_vision.evaluation.pipeline.base import EncoderDecoder


def annexPredictions(  # noqa: C901
    predictor=None,
    fo_dataset: Dataset = None,
    gt_field: str = "detections",
    predictor_field: str = "detectron-predictions",
    encoder_decoder=None,  # compressai_vision.evaluation.pipeline.base.EncoderDecoder
    use_pb: bool = False,  # progressbar.  captures stdion
    use_print: int = 1,  # print progress at each n:th line.  good for batch jobs
):
    """Run detector and EncoderDecoder instance on a dataset.  Append detector results and bits-per-pixel to each sample.

    :param predictor: A Detectron2 predictor
    :param fo_dataset: Fiftyone dataset
    :param gt_field: Which dataset member to use for ground truths.  Default: "detections"
    :param predictor_field: Which dataset member to use for saving the Detectron2 results.  Default: "detectron-predictions"
    :param encoder_decoder: (optional) a ``compressai_vision.evaluation.pipeline.EncoderDecoder`` subclass instance to apply on the image before detection
    :param use_pb: Show progressbar or not.  Nice for interactive runs, not so much for batch jobs.  Default: False.
    :param use_print: Print progress at every n:th. step.  Default: 0 = no printing.
    """
    assert predictor is not None, "provide Detectron2 predictor"
    assert fo_dataset is not None, "provide fiftyone dataset"
    if encoder_decoder is not None:
        assert issubclass(
            encoder_decoder.__class__, EncoderDecoder
        ), "encoder_decoder instances needs to be a subclass of EncoderDecoder"

    model_meta = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])

    """we don't need this!
    d2_dataset = FO2DetectronDataset(
        fo_dataset=fo_dataset,
        detection_field=detection_field,
        model_catids = model_meta.things_classes,
        )
    """
    try:
        _ = findLabels(fo_dataset, detection_field=gt_field)
    except ValueError:
        print(
            "your ground truths are empty: samples have no member '",
            gt_field,
            "' will set allowed_labels to empty list",
        )
        # allowed_labels = []

    # use open image ids if avail
    if fo_dataset.get_field("open_images_id"):
        id_field_name = "open_images_id"
    else:
        id_field_name = "id"

    npix_sum = 0
    nbits_sum = 0
    cc = 0
    # with ProgressBar(fo_dataset) as pb: # captures stdout
    if use_pb:
        pb = ProgressBar(fo_dataset)
    for sample in fo_dataset:
        cc += 1
        # sample.filepath
        path = sample.filepath
        im = cv2.imread(path)
        if im is None:
            print("FATAL: could not read the image file '" + path + "'")
            return -1
        # tag = path.split(os.path.sep)[-1].split(".")[0]  # i.e.: /path/to/some.jpg --> some.jpg --> some
        # if open_images_id is avail, then use it, otherwise use normal id
        tag = sample[id_field_name]
        if encoder_decoder is not None:
            # before using a detector, crunch through
            # encoder/decoder
            try:
                nbits, im_ = encoder_decoder.BGR(
                    im, tag=tag
                )  # include a tag for cases where EncoderDecoder uses caching
            except Exception as e:
                print("EncoderDecoder failed with '" + str(e) + "'")
                print("Traceback:")
                traceback.print_exc()
                return -1
            if nbits < 0:
                # there's something wrong with the encoder/decoder process
                # say, corrupt data from the VTMEncode bitstream etc.
                print("EncoderDecoder returned error: will try using it once again")
                nbits, im_ = encoder_decoder.BGR(im, tag=tag)
            if nbits < 0:
                print("EncoderDecoder returned error - again!  Will abort calculation")
                return -1

            # NOTE: use tranformed image im_
            npix_sum += im_.shape[0] * im_.shape[1]
            nbits_sum += nbits
        else:
            im_ = im

        res = predictor(im_)

        predictions = detectron251(
            res,
            model_catids=model_meta.thing_classes,
            # allowed_labels=allowed_labels # not needed, really
        )  # --> fiftyone Detections object

        """# could save nbits into each sample:
        if encoder_decoder is not None:
            predictions.nbits = nbits
        """
        sample[predictor_field] = predictions

        sample.save()
        if use_pb:
            pb.update()
        # print(">>>", cc%use_print)
        if use_print > 0 and ((cc % use_print) == 0):
            print("sample: ", cc, "/", len(fo_dataset))
    if use_pb:
        pb.close()

    # calculate bpp as defined by the VCM working group:
    bpp = None
    if encoder_decoder:
        if npix_sum < 1:
            print("error: number of pixels sum < 1")
            return -1
        if nbits_sum < 1:
            print("error: number of bits sum < 1")
            return -1
        bpp = nbits_sum / npix_sum
    return bpp


def annexVideoPredictions(  # noqa: C901
    predictor=None,
    fo_dataset: Dataset = None,  # video dataset
    gt_field: str = "detections",
    predictor_field: str = "detectron-predictions",
    encoder_decoder=None,  # compressai_vision.evaluation.pipeline.base.EncoderDecoder
    use_pb: bool = False,  # progressbar.  captures stdion
    use_print: int = 1,  # print progress at each n:th line.  good for batch jobs
):
    """Run detector and EncoderDecoder instance on a dataset.  Append detector results and bits-per-pixel to each sample.

    Dataset.Sample.Frames

    :param predictor: A Detectron2 predictor
    :param fo_dataset: A fiftyone video dataset
    :param gt_field: Which dataset member to use for ground truths.  Default: "detections"
    :param predictor_field: Which dataset member to use for saving the Detectron2 results.  Default: "detectron-predictions"
    :param encoder_decoder: (optional) a ``compressai_vision.evaluation.pipeline.EncoderDecoder`` subclass instance to apply on the image before detection
    :param use_pb: Show progressbar or not.  Nice for interactive runs, not so much for batch jobs.  Default: False.
    :param use_print: Print progress at every n:th. step.  Default: 0 = no printing.


    Video datasets look like this:


    ::

        Name:        sfu-hw-objects-v1
        Media type:  video
        Num samples: 1
        Persistent:  True
        Tags:        []
        Sample fields:
            id:         fiftyone.core.fields.ObjectIdField
            filepath:   fiftyone.core.fields.StringField
            tags:       fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
            metadata:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.VideoMetadata)
            media_type: fiftyone.core.fields.StringField
            class_tag:  fiftyone.core.fields.StringField
            name_tag:   fiftyone.core.fields.StringField
        Frame fields:
            id:           fiftyone.core.fields.ObjectIdField
            frame_number: fiftyone.core.fields.FrameNumberField
            detections:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)


    Difference between image & video datasets

    Image dataset:

    ::

        Dataset
            first()
            __index__()
            iterator
            --> return Sample objects
                fields: id, filepath, ground-truths, detections, etc.


    Video dataset

    ::

        Dataset
            first()
            __index__
            iterator
            --> return Sample objects
                fields: id, filepath
                frames: Frames object
                --> __index__
                    iterator
                    --> returns Frame object
                    fields: id, ground-truths, detections, etc.

    """
    assert predictor is not None, "provide Detectron2 predictor"
    assert fo_dataset is not None, "provide fiftyone dataset"
    if encoder_decoder is not None:
        assert issubclass(
            encoder_decoder.__class__, EncoderDecoder
        ), "encoder_decoder instances needs to be a subclass of EncoderDecoder"

    model_meta = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])

    try:
        _ = findVideoLabels(fo_dataset, detection_field=gt_field)
    except ValueError:
        print(
            "your ground truths are empty: samples have no member '",
            gt_field,
            "' will set allowed_labels to empty list",
        )
        # allowed_labels = []

    # use custom id field if avail
    if fo_dataset.get_field("custom_id"):
        id_field_name = "custom_id"
    else:
        id_field_name = "id"

    # LOOP OVER SAMPLES / START
    for sample in fo_dataset:
        print("USING VIDEO", sample.filepath)
        if use_pb:
            pb = ProgressBar(len(sample.frames))
        # read video!
        vid = cv2.VideoCapture(sample.filepath)
        npix_sum = 0
        nbits_sum = 0
        cc = 0
        # ITERATE OVER FRAMES / START
        for (
            n_frame
        ) in sample.frames:  # the iterator spits out frame numbers, nothing else
            cc += 1
            frame = sample.frames[n_frame]  # Frame object
            if frame.id is None:
                # ghost frames are created if you do sample.frames[num] with non-existent frame numbers (!)
                # https://github.com/voxel51/fiftyone/issues/2238
                print(
                    "void frame in fiftyone video dataset at frame number",
                    n_frame,
                    "will skip",
                )
                continue

            n_frame_file = (
                int(vid.get(cv2.CAP_PROP_POS_FRAMES)) + 1
            )  # next frame number from next call to vid.read()
            # print(frame.frame_number, n_frame_file)

            if frame.frame_number != n_frame_file:
                print("seeking to", frame.frame_number)
                ok = vid.set(cv2.CAP_PROP_POS_FRAMES, frame.frame_number - 1)
                if not ok:
                    vid.release()
                    print("could not seek to", frame.frame_number - 1)
                    return -1

            ok, arr = vid.read()
            if not ok:
                print("read failed at", frame.frame_number, "will try again")
                print("seeking to", frame.frame_number)
                ok = vid.set(cv2.CAP_PROP_POS_FRAMES, frame.frame_number - 1)
                if not ok:
                    vid.release()
                    print("could not seek to", frame.frame_number)
                    return -1
                ok, arr = vid.read()
                if not ok:
                    vid.release()
                    print("could not read video at frame", frame.frame_number)
                    return -1

            im = arr
            if im is None:
                print("FATAL: error reading video: got None array")
                vid.release()
                return -1

            # a unique tag for this frame image consists of video sample tag and the frame number
            tag = sample[id_field_name] + "_" + str(frame.frame_number)

            if encoder_decoder is not None:
                # before using a detector, crunch through
                # encoder/decoder
                try:
                    nbits, im_ = encoder_decoder.BGR(
                        im, tag=tag
                    )  # include a tag for cases where EncoderDecoder uses caching
                except Exception as e:
                    print("EncoderDecoder failed with '" + str(e) + "'")
                    print("Traceback:")
                    traceback.print_exc()
                    vid.release()
                    return -1
                if nbits < 0:
                    # there's something wrong with the encoder/decoder process
                    # say, corrupt data from the VTMEncode bitstream etc.
                    print("EncoderDecoder returned error: will try using it once again")
                    nbits, im_ = encoder_decoder.BGR(im, tag=tag)
                if nbits < 0:
                    print(
                        "EncoderDecoder returned error - again!  Will abort calculation"
                    )
                    vid.release()
                    return -1
                # NOTE: use tranformed image im_
                npix_sum += im_.shape[0] * im_.shape[1]
                nbits_sum += nbits
            else:
                im_ = im

            res = predictor(im_)

            predictions = detectron251(
                res,
                model_catids=model_meta.thing_classes,
                # allowed_labels=allowed_labels # not needed, really
            )  # --> fiftyone Detections object

            """# could save nbits into each sample:
            if encoder_decoder is not None:
                predictions.nbits = nbits
            """
            frame[predictor_field] = predictions

            frame.save()
            if use_pb:
                pb.update()
            # print(">>>", cc%use_print)
            if use_print > 0 and ((cc % use_print) == 0):
                print("frame: ", cc, "/", len(sample.frames), "of", sample.filepath)
        # ITERATE OVER FRAMES / STOP
        vid.release()
        if use_pb:
            pb.close()

        if encoder_decoder:  # alert user if EncoderDecoder class was requested
            if npix_sum < 1:
                print("error: number of pixels sum < 1 for video", sample.filepath)
            if nbits_sum < 1:
                print("error: number of bits sum < 1  for video", sample.filepath)
            sample["nbits_sum"] = nbits_sum
            sample["npix_sum"] = npix_sum
            sample.save()
    # LOOP OVER SAMPLES / STOP

    # calculate final bpp
    # this can be calculated separately as well as we save the numbers
    # into the sample
    bpp = None
    if encoder_decoder:
        npix_grand_sum = 0
        nbits_grand_sum = 0
        cc = 0
        for sample in fo_dataset:
            cc += 1
            nbits_sum = sample["nbits_sum"]
            npix_sum = sample["npix_sum"]
            if npix_sum < 1 or nbits_sum < 1:
                print("WARNING: For bpp calculation, skipping video", sample.filepath)
                continue
            npix_grand_sum += npix_sum
            nbits_grand_sum += nbits_sum
        bpp = nbits_grand_sum / npix_grand_sum
    return bpp
