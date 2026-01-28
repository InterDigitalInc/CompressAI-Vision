import operator

from functools import reduce

import torch
import torch.nn as nn

from ptflops import get_model_complexity_info


def calc_complexity_nn_part1_dn53(vision_model, img):
    device = torch.device(vision_model.device)

    # input pre-processing
    img = img[0]["image"].unsqueeze(0).to(device)

    # backbone
    partial_model = vision_model.darknet
    kmacs, _ = measure_mac(
        partial_model=partial_model,
        input_res=(img, vision_model.features_at_splits, True),
        input_constructor=prepare_jde_darknet_input,
    )

    pixels = reduce(operator.mul, [p_size for p_size in img.shape])

    return kmacs, pixels


def calc_complexity_nn_part2_dn53(vision_model, dec_features):
    assert "data" in dec_features
    device = vision_model.device

    if isinstance(dec_features["data"][0], list):  # image task
        # x = {k: v[0] for k, v in x.items()}
        pass
    else:  # video task
        x = dec_features["data"]
        x = {
            k: v.unsqueeze(0).to(device=device)
            for k, v in zip(vision_model.split_layer_list, x.values())
        }

    # nn part2 - part1 (darknet)
    partial_model = vision_model.darknet
    kmacs, _ = measure_mac(
        partial_model=partial_model,
        input_res=(None, x, False),
        input_constructor=prepare_jde_darknet_input,
    )

    pixels = sum(
        [reduce(operator.mul, [p_size for p_size in d.shape]) for d in x.values()]
    )

    return kmacs, pixels


def calc_complexity_nn_part1_plyr(vision_model, img):
    # input pre-processing
    imgs = vision_model.model.preprocess_image(img)
    _, C, H, W = imgs.tensor.shape

    # backbone
    partial_model = vision_model.backbone

    kmacs, _ = measure_mac(
        partial_model=partial_model, input_res=(C, H, W), input_constructor=None
    )

    pixels = reduce(operator.mul, [p_size for p_size in imgs.tensor.shape])

    return kmacs, pixels


def calc_complexity_nn_part2_plyr(vision_model, dec_features, data):
    if isinstance(data[0], list):  # image task
        data = {k: v[0] for k, v in data.items()}

    device = vision_model.device

    input_res_list, partial_model_lst, input_constructure_lst = [], [], []

    # top block
    C, H, W = data[len(data) - 1].shape  # p5 shape
    input_res_list.append((C, H, W))
    partial_model_lst.append(vision_model.top_block)
    input_constructure_lst.append(None)

    # for proposal generator
    C, H, W = data[0].shape  # p2 shape
    cdummy = dummy(dec_features["input_size"])
    input_res_list.append((cdummy, (1, C, H, W), device))
    partial_model_lst.append(vision_model.proposal_generator)
    input_constructure_lst.append(prepare_proposal_input_fpn)

    # for roi head
    feature_pyramid = {f"p{k + 2}": v.to(device) for k, v in data.items()}
    feature_pyramid.update({"p6": vision_model.top_block(feature_pyramid["p5"])[0]})
    feature_pyramid = {k: v.unsqueeze(0) for k, v in feature_pyramid.items()}
    proposals, _ = vision_model.proposal_generator(cdummy, feature_pyramid, None)
    input_res_list.append((cdummy, (1, C, H, W), proposals, device))
    partial_model_lst.append(vision_model.roi_heads)
    input_constructure_lst.append(prepare_roi_head_input_fpn)

    kmacs_sum = 0
    for partial_model, input_res, input_constructure in zip(
        partial_model_lst, input_res_list, input_constructure_lst
    ):
        kmacs, _ = measure_mac(
            partial_model=partial_model,
            input_res=input_res,
            input_constructor=input_constructure,
        )

        kmacs_sum = kmacs_sum + kmacs

    pixels = sum(
        [reduce(operator.mul, [p_size for p_size in d.shape]) for d in data.values()]
    )

    return kmacs_sum, pixels


def measure_mac(partial_model, input_res, input_constructor):
    macs, params = get_model_complexity_info(
        partial_model,
        input_res=input_res,
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs / 1_000, params


class dummy:
    def __init__(self, img_size: list):
        self.image_sizes = img_size


def get_downsampled_shape(h, w, ratio):
    import math

    n = int(math.log2(ratio))
    for _ in range(n):
        h, w = (h + 1) // 2, (w + 1) // 2
    return h, w


class YoloxPart1(nn.Module):
    def __init__(self, vision_model, split_id):
        super().__init__()
        self.backbone = vision_model.backbone
        self.split_id = split_id
        self.squeeze_at_split_enabled = vision_model.squeeze_at_split_enabled
        if self.squeeze_at_split_enabled:
            self.squeeze_model = vision_model.squeeze_model

    def forward(self, x):
        if self.split_id == "l13":
            y = self.backbone.stem(x)
            y = self.backbone.dark2(y)
            y = self.backbone.dark3[0](y)
            if self.squeeze_at_split_enabled:
                y = self.squeeze_model.squeeze_(y)
        elif self.split_id == "l37":
            y = self.backbone.stem(x)
            y = self.backbone.dark2(y)
            y = self.backbone.dark3(y)
        return y


class YoloxPart2(nn.Module):
    def __init__(self, vision_model, split_id):
        super().__init__()
        self.backbone = vision_model.backbone
        self.out1_cbl = vision_model.yolo_fpn.out1_cbl
        self.out1 = vision_model.yolo_fpn.out1
        self.out2_cbl = vision_model.yolo_fpn.out2_cbl
        self.out2 = vision_model.yolo_fpn.out2
        self.upsample = vision_model.yolo_fpn.upsample
        self.head = vision_model.head
        self.split_id = split_id
        self.squeeze_at_split_enabled = vision_model.squeeze_at_split_enabled
        if self.squeeze_at_split_enabled:
            self.squeeze_model = vision_model.squeeze_model
        # self.postprocess = vision_model.postprocess # Not needed for MAC calc

    def forward(self, x):
        y = x
        if self.split_id == "l13":
            if self.squeeze_at_split_enabled:
                y = self.squeeze_model.expand_(y)
            for proc_module in self.backbone.dark3[1:]:
                y = proc_module(y)

        fp_lvl2 = y
        fp_lvl1 = self.backbone.dark4(fp_lvl2)
        fp_lvl0 = self.backbone.dark5(fp_lvl1)

        # yolo branch 1
        b1_in = self.out1_cbl(fp_lvl0)
        b1_in = self.upsample(b1_in)
        b1_in = torch.cat([b1_in, fp_lvl1], 1)
        fp_lvl1 = self.out1(b1_in)

        # yolo branch 2
        b2_in = self.out2_cbl(fp_lvl1)
        b2_in = self.upsample(b2_in)
        b2_in = torch.cat([b2_in, fp_lvl2], 1)
        fp_lvl2 = self.out2(b2_in)

        outputs = self.head((fp_lvl2, fp_lvl1, fp_lvl0))
        return outputs


def calc_complexity_nn_part1_yolox(vision_model, img):
    device = torch.device(vision_model.device)
    img = img[0]["image"].unsqueeze(0).to(device)

    partial_model = YoloxPart1(vision_model, vision_model.split_id)

    C, H, W = img.shape[1:]

    kmacs, _ = measure_mac(
        partial_model=partial_model,
        input_res=(C, H, W),
        input_constructor=None,
    )

    pixels = reduce(operator.mul, [p_size for p_size in img.shape])
    return kmacs, pixels


def calc_complexity_nn_part2_yolox(vision_model, dec_features):
    assert "data" in dec_features

    x_data = dec_features["data"]

    x_data = {
        k: (v[0] if isinstance(x_data[0], list) else v).to(vision_model.device)
        for k, v in zip(vision_model.split_layer_list, x_data.values())
    }

    input_tensor = x_data[vision_model.split_id]

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    C, H, W = input_tensor.shape[1:]
    partial_model = YoloxPart2(vision_model, vision_model.split_id)

    kmacs, _ = measure_mac(
        partial_model=partial_model,
        input_res=(C, H, W),
        input_constructor=None,
    )

    pixels = reduce(operator.mul, input_tensor.shape)

    return kmacs, pixels


def prepare_proposal_input_fpn(resolutions):
    b, c, h, w = resolutions[1]
    resized_img = resolutions[0]
    device = resolutions[2]
    feature_lst = [torch.FloatTensor(*resolutions[1]).to(device)]
    feature_shape = [feature_lst[0].shape]
    for i in range(4):
        b, c, h, w = feature_shape[i]
        feature_shape.append((b, c, *get_downsampled_shape(h, w, 2)))
        feature_lst.append(torch.FloatTensor(*feature_shape[-1]).to(device))

    feature_dict = {f"p{e + 2}": feature for e, feature in enumerate(feature_lst)}

    return dict(images=resized_img, features=feature_dict, gt_instances=None)


def prepare_roi_head_input_fpn(resolutions):
    b, c, h, w = resolutions[1]
    resized_img = resolutions[0]
    proposals = resolutions[2]
    device = resolutions[3]
    feature_lst = [torch.FloatTensor(*resolutions[1]).to(device)]
    feature_shape = [feature_lst[0].shape]
    for i in range(4):
        b, c, h, w = feature_shape[i]
        feature_shape.append((b, c, *get_downsampled_shape(h, w, 2)))
        feature_lst.append(torch.FloatTensor(*feature_shape[-1]).to(device))

    feature_dict = {f"p{e + 2}": feature for e, feature in enumerate(feature_lst)}
    return dict(
        images=resized_img, features=feature_dict, proposals=proposals, targets=None
    )


def prepare_jde_darknet_input(resolutions):
    img_size = resolutions[0]
    feature_at_splits = resolutions[1]
    is_nn_part1 = resolutions[2]

    return dict(x=img_size, splits=feature_at_splits, is_nn_part1=is_nn_part1)


def prepare_jde_jdeprocess_input(resolutions):
    x = resolutions[0]
    return dict(x=x)
