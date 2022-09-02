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

import copy

import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.instances import Instances

# from collections import OrderedDict


def mapInputDict(mapper: dict = None, input: dict = None, verbose=False):
    """mapping detectron inputs dict from one class id to another

    :param mapper: a dictionary with key & value class id numbers
    :param input: a detectron2 input dict, i.e.:

    ::

        {'file_name': '001464cfae2a30b8-2.jpg',
            'height': 683,
            'width': 1024,
            'image_id': 1,
            'annotations': [{'iscrowd': 0,
                'bbox': [0.0, 238.74778616, 842.76105216, 444.25221383999997],
                'category_id': 3,
                'bbox_mode': <BoxMode.XYWH_ABS: 1>},
                  ... }
                ...
            ]
        }

    """
    assert mapper is not None
    assert input is not None
    output = copy.deepcopy(input)
    annotations_ = []  # new list of annotations
    for annotation in output["annotations"]:
        old_i = annotation["category_id"]  # old annotation class id
        try:
            new_i = mapper[old_i]  # let's see if we can map that
        except KeyError:  # no luck, so just scrap that annotations
            if verbose:
                print("scrapping annotation from category", old_i)
        else:
            if verbose:
                print("mapping annotation from category", old_i, "to", new_i)
            annotation["category_id"] = new_i
            annotations_.append(annotation)
    output["annotations"] = annotations_
    # replace old annotations list with new one
    return output


def mapInstances(mapper: dict, instances: Instances, verbose=False) -> Instances:
    """mapping detectron output Instance objects from one class id to another

    :param mapper: a dictionary with key=class id from the detector.
        value=corresponding class id in the ground truth set.
    :param instances: a detectron2.structures.instances.Instances instance that
        will be modified/mapped

    An example.  Assume the following mapper:

    ::

        {
            70: 1, 71: 2, 74: 3
        }

    - Detector prediction class id 70 is mapped to ground truth dataset class 1
    - Detector prediction class id 71 is mapped to ground truth dataset class 1
    - Detector prediction class id 69 is removed, since it is not in the keys
    """
    pick = []
    mapto = []
    for n, i in enumerate(instances.get("pred_classes")):
        i_ = i.item()
        try:
            m = mapper[i_]
        except KeyError:
            if verbose:
                print("scrapping input class", i_)
        else:
            if verbose:
                print("mapping input", i_, "to output", m)
            pick.append(n)
            mapto.append(m)
    if verbose:
        print("picking indexes", pick)
        print("they are mapped to", mapto)
    mapped_instances = instances[pick]  # pick classes
    mapped_instances.set(
        "pred_classes", torch.tensor(mapto, dtype=torch.long)
    )  # map classes
    return mapped_instances


def mapDataset(
    mapper: dict = None,
    source_name=None,
    target_name=None,
    base_name=None,
    verbose=False,
    use_voids=True,
):
    """Map dataset class indexes to another set of class indexes

    :param source_name:     name of source dataset
    :param target_name:     name target/mapped dataset
    :param base_name:       name of original dataset the detector was trained
              with .. we get the class names from here


    f.e. suppose this mapping

    ::

        gt          detector

        t2,m2  ---> t1,m1

        0 Book -->  73 book 1 Chair -->  56 chair 2 Clock -->  74 clock 3 Dining
        table -->  60 dining table 4 Microwave -->  68 microwave 5 Person -->  0
        person 6 Potted plant -->  58 potted plant 7 Refrigerator -->  72
        refrigerator 8 Tv -->  62 tv 9 Vase -->  75 vase ... 20 Humanoid --> 0
        person  # NOTE: we can also have many-to-one mappings ...


        map detector ids --> gt ids
            - detector spits out instances .. map those instances to gt
              instances
            - ..remove instances not compatible with gt
            - gt has been pruned
            - compare


        map gt ids --> detector ids
            - whole gt dataset has been premapped to detectors ids
            - detector spits out instances .. remove instances that are not in
              gt


        detector spits out 81 .. but that's not in t3 gt mapped to detector
        class numbers ..

    """
    assert source_name is not None
    assert target_name is not None
    assert base_name is not None

    if target_name in list(DatasetCatalog):
        print("WARNING: will re-register")
        DatasetCatalog.remove(target_name)
        MetadataCatalog.remove(target_name)

    ds = DatasetCatalog.get(source_name)  # source: list of dict(s)
    # mt = MetadataCatalog.get(source_name)  # source: metadata
    # base_meta = MetadataCatalog.get(base_name)
    # the class ids are taken from here
    new_ds = []
    for sample in ds:
        new_sample = mapInputDict(mapper=mapper, input=sample)
        new_ds.append(new_sample)

    def func():
        # print("reading func")
        return new_ds
        return []

    DatasetCatalog.register(target_name, func)

    lis = MetadataCatalog.get(base_name).thing_classes  # original tags
    existing_classes = list(mapper.values())
    if verbose:
        print("existing class indices in mapped dataset", existing_classes)
    newlis = []
    for i, tag in enumerate(lis):
        if verbose:
            print("original index", i)
        if i in existing_classes:
            newlis.append(tag)
        else:
            if use_voids:
                newlis.append("<void>")
            else:
                newlis.append(tag)
    if verbose:
        print("new tags", newlis)
    MetadataCatalog.get(target_name).thing_classes = newlis

    # copy remaining metadata from base dataset
    for key in MetadataCatalog.get(base_name).as_dict().keys():
        # MetadataCatalog.get(target_name).attribute = some_value
        if key not in ["name", "thing_classes", "json_file"]:
            setattr(
                MetadataCatalog.get(target_name),
                key,
                getattr(MetadataCatalog.get(base_name), key),
            )


def filterInstances(instances: Instances = None, lis: list = None) -> Instances:
    """
    :param instances: a Detectron2 Instances instance
    :param lis: class ids to be picked from the results
    """
    pick = []
    for n, i in enumerate(instances.get("pred_classes")):
        # n: index
        # i: class id
        i_ = i.item()
        if i_ in lis:
            pick.append(n)
    filtered_instances = instances[pick]  # Instance objects use indexing..
    # print("original instances", instances.get("pred_classes"))
    # print("picked instances", filtered_instances.get("pred_classes"))
    return filtered_instances
