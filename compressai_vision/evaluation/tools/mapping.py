from collections import OrderedDict


def findMapping(det: list = None, gt: list = None):
    """
    :param det: A list of tags, used by the trained detector i.e. ["cat, "dog", "horse", "plant", ..]
    :param gt: List of tags used by the evaluation ground truth set .. is different to det, but however has a notable
    intersection with the det list of tags

    valid/usable tags are the intersection of det & gt

    returns valid_tags: list, mapping: dict

    mapping is a mapping from ground truth class ids to the class ids used by the detector

    An example:

    ::

        det = ["cat", "dog", "horse", "plant"]
        gt = ["Car", "Cat", "Dog"]
        ==> mapping = {1:0, 2:1} & valid_tags = ["cat", "dog"]

    Supposing ``train_classes`` & ``gt_classes`` are lists of tags, then:

    ::

        for m1, m2 in mapping.items(): # key, value
            print(m1, gt_classes[m1], "--> ", m2, target_classes[m2])

    gives this output:

    ::

        1 Cat -->  0 cat
        2 Dog -->  1 dog

    You can use mapDataset to go from gt format dataset to detector formatted dataset
    """
    assert det is not None
    assert gt is not None

    # all tags to lowercase
    det = [l.lower() for l in det]
    gt = [l.lower() for l in gt]
    d = OrderedDict()
    ins = set(det).intersection(set(gt))  # {'cat','dog'}
    lis = list(ins)  # ['cat','dog']
    tags = []
    indexes = []

    for l in lis:  # tags intersection
        tags.append(l)
        val = det.index(l)
        key = gt.index(l)
        d[key] = val
        indexes.append(key)
    indexes.sort()
    d_sorted = OrderedDict()
    for i in indexes:
        d_sorted[i] = d[i]
    return tags, d_sorted
