import operator

from functools import reduce

import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import conv_flop_jit, get_shape, prod


def calc_complexity_nn_part1_dn53(vision_model, img):
    device = torch.device(vision_model.device)

    # input pre-processing
    img = img[0]["image"].unsqueeze(0).to(device)

    # backbone
    partial_model = DarknetBackboneOnlyFvcoreWrapper(
        vision_model.darknet,
        vision_model.features_at_splits,
        is_nn_part1=True,
    ).eval()
    kmacs = measure_kmacs(partial_model, img)

    pixels = reduce(operator.mul, [p_size for p_size in img.shape])

    return kmacs, pixels


def calc_complexity_nn_part2_dn53(vision_model, dec_features):
    assert "data" in dec_features

    if isinstance(dec_features["data"][0], list):  # image task
        # x = {k: v[0] for k, v in x.items()}
        # If image-task path exists, keep the same behavior as your original code.
        # You can implement the mapping here if needed.
        raise NotImplementedError(
            "Image-task path is not implemented yet for DN53 complexity."
        )

    else:  # video task
        x = dec_features["data"]

    # NN-part2 (Darknet backbone only): store features in wrapper, pass tensor-only dummy input to fvcore
    partial_model = DarknetNNPart2BackboneOnlyFvcoreWrapper(
        vision_model.darknet, vision_model.features_at_splits
    ).eval()

    # fvcore input must be Tensor (wrapper overwrites internal 'x' from stored features)
    x_dummy = next(iter(x.values()))
    # fvcore input must be a Tensor; pick a deterministic dummy tensor

    kmacs = measure_kmacs(partial_model, x_dummy)

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
    kmacs = measure_kmacs(partial_model, imgs.tensor)

    pixels = reduce(operator.mul, [p_size for p_size in imgs.tensor.shape])

    return kmacs, pixels


def calc_complexity_nn_part2_plyr(vision_model, dec_features, data):
    # Special handling for image task (keep existing logic)
    if isinstance(next(iter(data.values())), list):
        data = {k: v[0] for k, v in data.items()}

    device = vision_model.device

    # 1) Build feature pyramid using actual decoded features
    # Convert each feature from (C, H, W) to (1, C, H, W)
    feature_pyramid = {
        f"p{k + 2}": v.to(device).unsqueeze(0) for k, v in data.items()
    }  # p2–p5

    # Generate p6 from p5 using top_block
    top_block_out = vision_model.top_block(feature_pyramid["p5"])
    p6 = top_block_out[0] if isinstance(top_block_out, (tuple, list)) else top_block_out
    feature_pyramid["p6"] = p6  # (1, C, H, W)

    # 2) Measure top_block FLOPs (p5 -> p6 only)
    kmacs_sum = 0.0
    kmacs_sum += measure_kmacs(vision_model.top_block, feature_pyramid["p5"])

    # 3) Measure RPN head only (exclude proposal post-processing)
    rpn_head_model = RPNHeadOnlyFvcoreWrapper(vision_model.proposal_generator).eval()
    kmacs_sum += measure_kmacs(
        rpn_head_model,
        (
            feature_pyramid["p2"],
            feature_pyramid["p3"],
            feature_pyramid["p4"],
            feature_pyramid["p5"],
            feature_pyramid["p6"],
        ),
    )

    # 4) Measure sem_seg_head if available
    # Panoptic/Semantic models use sem_seg_head(x, None)
    is_semseg = (
        hasattr(vision_model, "sem_seg_head") and vision_model.sem_seg_head is not None
    )
    if is_semseg:
        semseg_model = SemSegHeadFvcoreWrapper(vision_model.sem_seg_head).eval()
        # IMPORTANT: pass dict as a single positional arg
        kmacs_sum += measure_kmacs(semseg_model, (feature_pyramid,))

    #  5) ROIHeads
    # Run the proposal generator once to obtain actual proposals.
    # Only the image size is required for Detectron2, so a minimal dummy object is used.
    class _ImagesDummy:
        def __init__(self, image_sizes):
            self.image_sizes = image_sizes

    # Assume dec_features["input_size"] follows the same structure as the original pipeline
    images = _ImagesDummy(dec_features["input_size"])

    with torch.no_grad():
        proposals, _ = vision_model.proposal_generator(images, feature_pyramid, None)

    # 5-1) Measure box_head + box_predictor
    # ROIAlign/Pooler is excluded from FLOPs due to ambiguity and potential CUDA/JIT issues.
    # Instead, pooled features are obtained once and only NN blocks are measured.
    if hasattr(vision_model, "roi_heads") and vision_model.roi_heads is not None:
        roi_heads = vision_model.roi_heads

        # Follow roi_heads.in_features order if available
        if hasattr(roi_heads, "in_features"):
            in_feats = list(roi_heads.in_features)
        else:
            in_feats = ["p2", "p3", "p4", "p5"]

        feat_list = [feature_pyramid[f] for f in in_feats if f in feature_pyramid]

        # Convert proposals to the format required by the box_pooler
        # Detectron2 uses Instances.proposal_boxes by default
        boxes = [p.proposal_boxes for p in proposals]

        with torch.no_grad():
            pooled = roi_heads.box_pooler(
                feat_list, boxes
            )  # (num_boxes, C, pool_h, pool_w)

        box_head_model = BoxHeadPredictorFvcoreWrapper(roi_heads).eval()
        kmacs_sum += measure_kmacs(box_head_model, pooled)

        # 5-2) Measure mask head if available
        if (not is_semseg) and (
            hasattr(roi_heads, "mask_head")
            and roi_heads.mask_head is not None
            and hasattr(roi_heads, "mask_pooler")
        ):
            # Only run when mask task is actually enabled
            if sum(len(p) for p in proposals) > 0:
                # Run ROIHeads once to obtain pred_instances
                with torch.no_grad():
                    pred_instances, _ = roi_heads(
                        images, feature_pyramid, proposals, None
                    )

                # Skip if no detected objects
                if sum(len(p) for p in pred_instances) > 0:
                    # Mask pooling requires pred_boxes
                    mask_boxes = [p.pred_boxes for p in pred_instances]

                    with torch.no_grad():
                        mask_pooled = roi_heads.mask_pooler(feat_list, mask_boxes)

                    pred_classes = torch.cat([p.pred_classes for p in pred_instances])
                    mask_head_model = MaskHeadFvcoreWrapper(
                        roi_heads, pred_classes
                    ).eval()
                    kmacs_sum += measure_kmacs(mask_head_model, mask_pooled)

    pixels = sum([reduce(operator.mul, list(d.shape)) for d in data.values()])

    return kmacs_sum, pixels


def _flops_to_kmacs(total_flops: float) -> float:
    """
    Convert FLOPs reported by fvcore into KMACs.

    Note:
    - fvcore typically counts multiply and add operations separately.
    - Therefore, FLOPs are divided by 2 to approximate MACs.
    - The result is further scaled to kilo-MACs (KMACs).
    """
    return float(total_flops) / 1e3


# (mem-fix) KMACs are deterministic w.r.t. (module, input shapes), so trace each
# shape only once. Per-image FlopCountAnalysis (torch.jit tracing) retains feature-map
# tensors and steadily grows CPU RAM (OOM); caching avoids that and speeds up
# repeated measurement.
_KMACS_CACHE: dict = {}


def _shape_key(x):
    """Build a hashable shape key from inputs (possibly nested)."""
    if torch.is_tensor(x):
        return ("t", tuple(x.shape))
    if isinstance(x, (list, tuple)):
        return ("s", tuple(_shape_key(v) for v in x))
    if isinstance(x, dict):
        return ("d", tuple((k, _shape_key(v)) for k, v in x.items()))
    return ("o", repr(x))


def measure_kmacs(module: nn.Module, inputs, tag: str = None) -> float:
    """
    Measure KMACs for a given module using fvcore.

    This function:
    - Ensures the module is in evaluation mode.
    - Automatically casts inputs to the correct device and dtype.
    - Supports nested iterable inputs (tuple, list, dict).
    - Handles modules without trainable parameters.

    Args:
        module: Target neural network module.
        inputs: Input tensor or nested structure of tensors.
        tag: Optional name for logging.

    Returns:
        KMACs value as a float.
    """

    module = module.eval()

    try:
        p = next(module.parameters())
    except StopIteration:
        name = tag or module.__class__.__name__
        print(f"[INFO] No parameters found in {name}, MACs set to 0.")
        return 0.0

    # Safe casting (recursive)
    def _cast(x):
        if torch.is_tensor(x):
            return x.to(device=p.device, dtype=p.dtype)
        elif isinstance(x, (list, tuple)):
            return type(x)(_cast(v) for v in x)
        elif isinstance(x, dict):
            return {k: _cast(v) for k, v in x.items()}
        else:
            return x

    inputs = _cast(inputs)

    if torch.is_tensor(inputs):
        inputs = (inputs,)  # single input -> tuple
    elif not isinstance(inputs, tuple):
        inputs = (inputs,)  # safe fallback

    # (mem-fix) Shape-based cache lookup: skip re-tracing for the same (module, input shapes).
    cache_key = None
    try:
        cache_key = (module.__class__.__qualname__, id(p), _shape_key(inputs))
    except Exception:
        cache_key = None
    if cache_key is not None and cache_key in _KMACS_CACHE:
        return _KMACS_CACHE[cache_key]

    with torch.no_grad():
        flops = FlopCountAnalysis(module, inputs)
        flops.set_op_handle(
            **{
                # conv ops by conv_flop_jit
                "aten::conv2d": conv_flop_jit,
                "aten::_convolution": conv_flop_jit,
                "aten::cudnn_convolution": conv_flop_jit,
                # element-wise ops (out-of-place)
                "aten::add": elemwise_flop_jit,
                "aten::add_": elemwise_flop_jit,
                "aten::mul": elemwise_flop_jit,
                "aten::mul_": elemwise_flop_jit,
                "aten::exp": elemwise_flop_jit,
                "aten::clamp_min": elemwise_flop_jit,
                "aten::div": elemwise_flop_jit,
                "aten::abs": elemwise_flop_jit,
                "aten::reciprocal": elemwise_flop_jit,
                "aten::round": elemwise_flop_jit,
                "aten::leaky_relu": elemwise_flop_jit,
                # pooling
                "aten::max_pool2d": max_pool2d_flop_jit,
            }
        )
        total_flops = flops.total()

        del flops

    kmacs = _flops_to_kmacs(total_flops)
    name = tag or module.__class__.__name__
    # print(f"[INFO] {name}: KMACs = {kmacs}")
    if cache_key is not None:
        _KMACS_CACHE[cache_key] = kmacs
    return kmacs


class SemSegHeadFvcoreWrapper(nn.Module):
    def __init__(self, sem_seg_head: nn.Module):
        super().__init__()
        self.sem_seg_head = sem_seg_head

    def forward(self, x):
        # detectron2 style: returns (sem_seg_results, losses) or similar
        out = self.sem_seg_head(x, None)
        return out[0] if isinstance(out, (tuple, list)) else out


class RPNHeadOnlyFvcoreWrapper(nn.Module):
    """
    Wrapper for Detectron2 RPN to measure FLOPs only for the neural network part.

    This excludes proposal generation and post-processing steps
    such as Top-K selection, sorting, and NMS.

    Only the RPN head (convolution + classification + regression)
    is executed for FLOPs measurement.
    """

    def __init__(self, proposal_generator):
        super().__init__()
        self.pg = proposal_generator  # Detectron2 RPN module

    def forward(self, p2, p3, p4, p5, p6):
        feats = [p2, p3, p4, p5, p6]
        return self.pg.rpn_head(feats)  # Returns objectness logits and box deltas


class BoxHeadPredictorFvcoreWrapper(nn.Module):
    """
    Wrapper to measure FLOPs for ROI box head and predictor only.

    The full ROIHeads module is not executed to avoid non-NN components.
    The input is expected to be pooled box features.

    Input shape:
        (num_boxes, C, pool_h, pool_w)
    """

    def __init__(self, roi_heads):
        super().__init__()
        self.box_head = roi_heads.box_head
        self.box_predictor = roi_heads.box_predictor

    def forward(self, box_features):
        x = self.box_head(box_features)
        scores, deltas = self.box_predictor(x)
        return scores, deltas


class MaskHeadFvcoreWrapper(nn.Module):
    def __init__(self, roi_heads, pred_classes):
        super().__init__()
        self.mask_head = roi_heads.mask_head
        self.pred_classes = pred_classes

    def forward(self, mask_features):
        # simulate detectron2 mask inference
        return self.mask_head.layers(mask_features)


class DarknetBackboneOnlyFvcoreWrapper(nn.Module):
    """
    fvcore-friendly wrapper for Darknet that always returns a Tensor.

    - Runs the same module_list loop as Darknet.forward()
    - Skips 'yolo' heads (so you can measure backbone-only FLOPs)
    - Returns the last feature tensor `x` instead of detection output
      (prevents returning None when no yolo layers are executed)
    """

    def __init__(self, darknet: nn.Module, splits: dict, is_nn_part1: bool):
        super().__init__()
        self.darknet = darknet
        self.splits = splits
        self.is_nn_part1 = is_nn_part1

    def forward(self, x):
        # local aliases
        module_defs = self.darknet.module_defs
        module_list = self.darknet.module_list

        layer_outputs = []
        had_yolo = False

        if self.is_nn_part1:
            sidx = 0
            eidx = max(self.splits.keys()) + 1
            splits = self.splits
        else:
            features = self.splits.copy()
            max_id = max(features.keys())

            if max_id <= 74:
                sidx = max_id + 1
                for idx in range(0, sidx):
                    if idx not in features:
                        layer_outputs.append(None)
                    else:
                        x = features[idx]
                        layer_outputs.append(x)
            else:
                sidx = min(features.keys())

            eidx = len(module_list)
            splits = features  # reuse name for convenience

        for i, (module_def, module) in enumerate(
            zip(module_defs[sidx:eidx], module_list[sidx:eidx])
        ):
            nn_idx = i + sidx

            if not self.is_nn_part1:
                if nn_idx in splits:
                    x = splits[nn_idx]
                    layer_outputs.append(x)
                    splits.pop(nn_idx)
                    had_yolo = False
                    continue
                elif had_yolo is True and nn_idx < min(splits.keys()):
                    continue

            mtype = module_def["type"]

            if mtype in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif mtype == "route":
                layer_i = [int(v) for v in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[j] for j in layer_i], 1)

            elif mtype == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif mtype == "yolo":
                # IMPORTANT: skip yolo head so we only count backbone FLOPs
                had_yolo = True
                # keep x unchanged, just store it

            layer_outputs.append(x)

            if self.is_nn_part1:
                if nn_idx in self.splits:
                    self.splits[nn_idx] = x

        # Always return a tensor to make tracing stable
        return x


class DarknetNNPart2BackboneOnlyFvcoreWrapper(nn.Module):
    """
    fvcore-friendly wrapper for Darknet assuming is_nn_part1 == False only.

    - Measures FLOPs for the backbone path only (skips YOLO heads and post-processing).
    - Preserves Darknet's original nn-part2 feature injection logic.
    - Always returns a Tensor to avoid fvcore reporting 0 FLOPs due to None outputs.

    Inputs:
    - forward(x_dummy): A placeholder Tensor for fvcore tracing.
      NOTE: This tensor is NOT used for computation. We initialize `x` from injected features.
    """

    def __init__(self, darknet: nn.Module, features: dict):
        super().__init__()
        self.darknet = darknet
        self.features = features  # dict[int, Tensor]

        self.is_nn_part1 = False  # fixed for this wrapper

    def forward(self, x_dummy: torch.Tensor) -> torch.Tensor:
        # ---- Fixed assumption: is_nn_part1 is always False (nn-part2) ----
        module_defs = self.darknet.module_defs
        module_list = self.darknet.module_list

        # Working copy (same as original)
        features = self.features.copy()

        layer_outputs = []
        had_yolo = False

        max_id = max(features.keys())

        # Match original nn-part2 logic for sidx/eidx and pre-filling layer_outputs
        if max_id <= 74:
            sidx = max_id + 1

            # Pre-fill layer_outputs[0:sidx] with injected features or None
            # Also pick a valid initial x from the earliest available injected feature
            for idx in range(0, sidx):
                if idx not in features:
                    layer_outputs.append(None)
                else:
                    x = features[idx]
                    layer_outputs.append(x)
        else:
            sidx = min(features.keys())

        eidx = len(module_list)
        # IMPORTANT: do NOT start from x_dummy (can cause channel mismatch before injection)

        # Main loop (same structure as original, but assumes nn-part2 only)
        for i, (module_def, module) in enumerate(
            zip(module_defs[sidx:eidx], module_list[sidx:eidx])
        ):
            nn_idx = i + sidx

            # --- Feature injection (same as original) ---
            if nn_idx in features.keys():
                x = features[nn_idx]
                layer_outputs.append(x)
                features.pop(nn_idx)
                had_yolo = False
                continue
            elif (
                had_yolo is True and len(features) > 0 and nn_idx < min(features.keys())
            ):
                continue

            mtype = module_def["type"]
            if mtype in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif mtype == "route":
                layer_i = [int(v) for v in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[j] for j in layer_i], 1)

            elif mtype == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif mtype == "yolo":
                x = module[0](x, self.darknet.img_size)
                had_yolo = True
                # Keep x unchanged

            layer_outputs.append(x)

        # Always return a Tensor so fvcore can produce FLOPs stats
        return x


def elemwise_flop_jit(inputs, outputs):
    # outputs can be Tensor or tuple/list of Tensors
    out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return prod(get_shape(out))  # 1 flop per output element (approx.)


def max_pool2d_flop_jit(inputs, outputs):
    """
    Approximate FLOPs for max_pool2d.

    Convention:
    - For each output element, max-pool performs (kH*kW - 1) comparisons.
    - We count comparisons as 1 FLOP each (approx).
    """
    # aten::max_pool2d signature (typical):
    out_numel = _value_numel(outputs[0])
    if out_numel == 0:
        return 0

    k = _to_ivalue(inputs[1], default=None)  # could be int or (kH,kW) or list
    if isinstance(k, int):
        kH, kW = k, k
    elif isinstance(k, (list, tuple)) and len(k) == 2:
        kH, kW = int(k[0]), int(k[1])
    else:
        # Fallback: if kernel size is not statically available, assume 1x1
        kH, kW = 1, 1

    # comparisons per output = kH*kW - 1
    return int(out_numel) * max(int(kH) * int(kW) - 1, 0)


def _value_sizes(v):
    """
    Get static tensor sizes from torch._C.Value (JIT IR value).
    Returns a list like [N, C, H, W] or None if unknown.
    """
    try:
        t = v.type()
        if hasattr(t, "sizes") and t.sizes() is not None:
            return list(t.sizes())
    except Exception:
        pass
    return None


def _value_numel(v):
    sizes = _value_sizes(v)
    if not sizes or any(s is None for s in sizes):
        return 0
    n = 1
    for s in sizes:
        n *= int(s)
    return n


def _to_ivalue(v, default=None):
    """
    Try to materialize constant from torch._C.Value if it is a constant.
    Works for many prim::Constant-derived Values.
    """
    try:
        return v.toIValue()
    except Exception:
        return default


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

    kmacs = measure_kmacs(partial_model, img)

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

    kmacs = measure_kmacs(partial_model, input_tensor)

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
