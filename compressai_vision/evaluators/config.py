# pandaset category
pandaset_category = {
    1: "Smoke",
    2: "Exhaust",
    3: "Spray or rain",
    4: "Reflection",
    5: "Vegetation",
    6: "Ground",
    7: "Road",
    8: "Lane Line Marking",
    9: "Stop Line Marking",
    10: "Other Road Marking",
    11: "Sidewalk",
    12: "Driveway",
    13: "Car",
    14: "Pickup Truck",
    15: "Medium-sized Truck",
    16: "Semi-truck",
    17: "Towed Object",
    18: "Motorcycle",
    19: "Other Vehicle - Construction Vehicle",
    20: "Other Vehicle - Uncommon",
    21: "Other Vehicle - Pedicab",
    22: "Emergency Vehicle",
    23: "Bus",
    24: "Personal Mobility Device",
    25: "Motorized Scooter",
    26: "Bicycle",
    27: "Train",
    28: "Trolley",
    29: "Tram / Subway",
    30: "Pedestrian",
    31: "Pedestrian with Object",
    32: "Animals - Bird",
    33: "Animals - Other",
    34: "Pylons",
    35: "Road Barriers",
    36: "Signs",
    37: "Cones",
    38: "Construction Signs",
    39: "Temporary Construction Barriers",
    40: "Rolling Containers",
    41: "Building",
    42: "Other Static Object",
    0: "unlabeled",
}

# Object that appear frequently in the dataset
# 4 5 6 7 8 10 11 13 30 31 36 37 41 42

# COCO things categories
coco_things = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# COCO stuff categories
coco_stuff = {
    0: "things",
    1: "banner",
    2: "blanket",
    3: "bridge",
    4: "cardboard",
    5: "counter",
    6: "curtain",
    7: "door-stuff",
    8: "floor-wood",
    9: "flower",
    10: "fruit",
    11: "gravel",
    12: "house",
    13: "light",
    14: "mirror-stuff",
    15: "net",
    16: "pillow",
    17: "platform",
    18: "playingfield",
    19: "railroad",
    20: "river",
    21: "road",
    22: "roof",
    23: "sand",
    24: "sea",
    25: "shelf",
    26: "snow",
    27: "stairs",
    28: "tent",
    29: "towel",
    30: "wall-brick",
    31: "wall-stone",
    32: "wall-tile",
    33: "wall-wood",
    34: "water",
    35: "window-blind",
    36: "window",
    37: "tree",
    38: "fence",
    39: "ceiling",
    40: "sky",
    41: "cabinet",
    42: "table",
    43: "floor",
    44: "pavement",
    45: "mountain",
    46: "grass",
    47: "dirt",
    48: "paper",
    49: "food",
    50: "building",
    51: "rock",
    52: "wall",
    53: "rug",
    -1: "background",
}

# COCO stuff to pandaset
coco_stuff_to_pandaset = {
    -1: 0,  # Background -> unlabeled
    0: 0,  # code will handle this
    1: 42,  # banner -> other static object
    2: 0,  # blanket -> unlabeled
    3: 0,  # bridge -> unlabeled
    4: 0,  # cardboard -> unlabeled
    5: 0,  # counter -> unlabeled
    6: 0,  # curtain -> unlabeled
    7: 0,  # door-stuff -> unlabeled
    8: 41,  # floor-wood -> building
    9: 5,  # flower -> vegetation
    10: 5,  # fruit -> vegetation
    11: 6,  # gravel -> ground
    12: 41,  # house -> building
    13: 0,  # light -> unlabeled
    14: 4,  # mirror-stuff -> reflection
    15: 39,  # net -> temporary construction barriers
    16: 0,  # pillow -> unlabeled
    17: 0,  # platform -> unlabeled
    18: 6,  # playingfield -> ground
    19: 6,  # railroad -> ground
    20: 0,  # river -> ignored
    21: 7,  # road -> road
    22: 41,  # roof -> building
    23: 6,  # sand -> ground
    24: 0,  # sea -> unlabeled
    25: 42,  # shelf -> other static object
    26: 0,  # snow -> unlabeled
    27: 0,  # stairs -> unlabeled
    28: 42,  # tent -> other static object
    29: 0,  # towel -> unlabeled
    30: 41,  # wall-brick -> building
    31: 41,  # wall-stone -> building
    32: 41,  # wall-tile -> building
    33: 41,  # wall-wood -> building
    34: 0,  # water -> unlabeled
    35: 41,  # window-blind -> building
    36: 41,  # window -> building
    37: 5,  # tree -> vegetation
    38: 39,  # fence -> temporary construction barriers
    39: 0,  # ceiling -> unlabeled
    40: 0,  # sky -> unlabeled
    41: 42,  # cabinet -> other static object
    42: 42,  # table -> other static object
    43: 41,  # floor -> building
    44: 11,  # pavement -> sidewalk
    45: 0,  # mountain -> unlabeled
    46: 5,  # grass -> vegetation
    47: 6,  # dirt -> ground
    48: 0,  # paper -> unlabeled
    49: 0,  # food -> unlabeled
    50: 41,  # building -> building
    51: 0,  # rock -> unlabeled
    52: 41,  # wall -> building
    53: 0,  # rug -> unlabeled
}


# COCO things to pandaset
coco_things_to_pandaset = {
    0: 30,  # person -> pedestrian
    1: 26,  # bicycle -> bicycle
    2: 13,  # car -> car
    3: 18,  # motorcycle -> motorcycle
    4: 20,  # airplane -> other vehicle - uncommon
    5: 23,  # bus -> bus
    6: 27,  # train -> train
    7: 14,  # truck -> pickup truck
    8: 20,  # boat -> other vehicle - uncommon
    9: 42,  # traffic light -> other static object
    10: 42,  # fire hydrant -> other static object
    11: 36,  # stop sign -> signs
    12: 42,  # parking meter -> other static object
    13: 42,  # bench -> other static object
    14: 32,  # bird -> animals - bird
    15: 33,  # cat -> animals - other
    16: 33,  # dog -> animals - other
    17: 33,  # horse -> animals - other
    18: 33,  # sheep -> animals - other
    19: 33,  # cow -> animals - other
    20: 33,  # elephant -> animals - other
    21: 33,  # bear -> animals - other
    22: 33,  # zebra -> animals - other
    23: 33,  # giraffe -> animals - other
    24: 0,  # backpack -> unlabeled
    25: 0,  # umbrella -> unlabeled
    26: 0,  # handbag -> unlabeled
    27: 0,  # tie -> unlabeled
    28: 0,  # suitcase -> unlabeled
    29: 0,  # frisbee -> unlabeled
    30: 0,  # skis -> unlabeled
    31: 0,  # snowboard -> unlabeled
    32: 0,  # sports ball -> unlabeled
    33: 0,  # kite -> unlabeled
    34: 0,  # baseball bat -> unlabeled
    35: 0,  # baseball glove -> unlabeled
    36: 0,  # skateboard -> unlabeled
    37: 0,  # surfboard -> unlabeled
    38: 0,  # tennis racket -> unlabeled
    39: 0,  # bottle -> unlabeled
    40: 0,  # wine glass -> unlabeled
    41: 0,  # cup -> unlabeled
    42: 0,  # fork -> unlabeled
    43: 0,  # knife -> unlabeled
    44: 0,  # spoon -> unlabeled
    45: 0,  # bowl -> unlabeled
    46: 0,  # banana -> unlabeled
    47: 0,  # apple -> unlabeled
    48: 0,  # sandwich -> unlabeled
    49: 0,  # orange -> unlabeled
    50: 0,  # broccoli -> unlabeled
    51: 0,  # carrot -> unlabeled
    52: 0,  # hot dog -> unlabeled
    53: 0,  # pizza -> unlabeled
    54: 0,  # donut -> unlabeled
    55: 0,  # cake -> unlabeled
    56: 42,  # chair -> other static object
    57: 0,  # couch -> unlabeled
    58: 5,  # potted plant -> vegetation
    59: 0,  # bed -> unlabeled
    60: 42,  # dining table -> other static object
    61: 0,  # toilet -> unlabeled
    62: 0,  # tv -> unlabeled
    63: 0,  # laptop -> unlabeled
    64: 0,  # mouse -> unlabeled
    65: 0,  # remote -> unlabeled
    66: 0,  # keyboard -> unlabeled
    67: 0,  # cell phone -> unlabeled
    68: 0,  # microwave -> unlabeled
    69: 0,  # oven -> unlabeled
    70: 0,  # toaster -> unlabeled
    71: 0,  # sink -> unlabeled
    72: 0,  # refrigerator -> unlabeled
    73: 0,  # book -> unlabeled
    74: 0,  # clock -> unlabeled
    75: 0,  # vase -> unlabeled
    76: 0,  # scissors -> unlabeled
    77: 0,  # teddy bear -> unlabeled
    78: 0,  # hair drier -> unlabeled
    79: 0,
}  # toothbrush -> unlabeled

# Categories candidates for VCM evaluation
vcm_category = {
    0: "unlabeled",
    1: "vegetation",
    2: "ground",
    3: "road",
    4: "sidewalk",
    5: "car",
    6: "truck",
    7: "motorcycle",
    8: "bicycle",
    9: "other vehicle",
    10: "bus",
    11: "train",
    12: "pedestrian",
    13: "animals - bird",
    14: "animals - other",
    15: "signs",
    16: "road barriers",
    17: "building",
    18: "other static object",
}

# Ignore list of categories in vcm_category
# ignore_in_vcm_category = [0, 2, 9, 11, 13, 14, 15, 16, 18]

# Categories used for VCM evaluation
vcm_eval_category = [1, 3, 4, 5, 6, 7, 8, 10, 12, 17]


# Map the pandaset category to the vcm category
pandaset_to_vcm_category = {
    0: 0,  # unlabeled -> unlabeled
    1: 0,  # smoke -> unlabeled
    2: 5,  # exhaust -> car
    3: 0,  # spray or rain -> unlabeled
    4: 0,  # reflection -> unlabeled
    5: 1,  # vegetation -> vegetation
    6: 2,  # ground -> ground
    7: 3,  # road -> road
    8: 3,  # lane line marking -> road
    9: 3,  # stop line marking -> road
    10: 3,  # other road marking -> road
    11: 4,  # sidewalk -> sidewalk
    12: 4,  # driveway -> sidewalk
    13: 5,  # car -> car
    14: 6,  # pickup truck -> truck
    15: 6,  # medium-sized truck -> truck
    16: 6,  # semi-truck -> truck
    17: 5,  # towed object -> car
    18: 7,  # motorcycle -> motorcycle
    19: 9,  # other vehicle - construction vehicle -> other vehicle
    20: 9,  # other vehicle - uncommon -> other vehicle
    21: 9,  # other vehicle - pedicab -> other vehicle
    22: 9,  # emergency vehicle -> other vehicle
    23: 10,  # bus -> bus
    24: 7,  # personal mobility device -> motorcycle
    25: 7,  # motorized scooter -> motorcycle
    26: 8,  # bicycle -> bicycle
    27: 11,  # train -> train
    28: 0,  # trolley -> unlabeled
    29: 11,  # tram / subway -> train
    30: 12,  # pedestrian -> pedestrian
    31: 12,  # pedestrian with object -> pedestrian
    32: 13,  # animals - bird -> animals - bird
    33: 14,  # animals - other -> animals - other
    34: 16,  # pylons -> road barriers
    35: 16,  # road barriers -> road barriers
    36: 15,  # signs -> signs
    37: 16,  # cones -> road barriers
    38: 15,  # construction signs -> signs
    39: 16,  # temporary construction barriers -> road barriers
    40: 18,  # rolling containers -> other static object
    41: 17,  # building -> building
    42: 18,
}  # other static object -> other static object


# write the dictionary to txt file
def write_dict_to_csv(dict, file_name, original_dict, tartget_dict):
    with open(file_name, "w") as f:
        for key, value in dict.items():
            f.write("%s_%s\n" % (original_dict[key], tartget_dict[value]))


# write_dict_to_csv(pandaset_to_vcm_category, 'pandaset_to_vcm_category.csv', pandaset_category, vcm_category)
# write_dict_to_csv(coco_stuff_to_pandaset, 'coco_stuff_to_pandaset.csv', coco_stuff, pandaset_category)
# write_dict_to_csv(coco_things_to_pandaset, 'coco_things_to_pandaset.csv', coco_things, pandaset_category)
