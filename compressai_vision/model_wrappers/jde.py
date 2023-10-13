# Copyright (c) 2022-2023, InterDigital Communications, Inc
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

import logging
from pathlib import Path
from typing import Dict, List

import torch
from jde.models import Darknet
from jde.tracker import matching
from jde.tracker.basetrack import TrackState
from jde.tracker.multitracker import (
    STrack,
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from jde.utils.kalman_filter import KalmanFilter
from jde.utils.utils import non_max_suppression, scale_coords
from torch import Tensor

from compressai_vision.model_wrappers.utils import compute_frame_resolution
from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper
from .utils import tensor_to_tiled, tiled_to_tensor

__all__ = [
    "jde_1088x608",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


@register_vision_model("jde_1088x608")
class jde_1088x608(BaseWrapper):
    def __init__(self, device="cpu", **kwargs):
        super().__init__()

        self.device = device
        self.model_info = {
            "cfg": f"{root_path}/{kwargs['cfg']}",
            "weight": f"{root_path}/{kwargs['weight']}",
        }

        self.model_configs = {
            "iou_thres": float(kwargs["iou_thres"]),
            "conf_thres": float(kwargs["conf_thres"]),
            "nms_thres": float(kwargs["nms_thres"]),
            "min_box_area": int(kwargs["min_box_area"]),
            "track_buffer": int(kwargs["track_buffer"]),
            "frame_rate": float(kwargs["frame_rate"]),
        }
        self.max_time_on_hold = int(
            self.model_configs["frame_rate"] / 30.0 * self.model_configs["track_buffer"]
        )

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_layer_list = kwargs["splits"]
        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        self.darknet = Darknet(self.model_info["cfg"], device, nID=14455)
        self.darknet.load_state_dict(
            torch.load(self.model_info["weight"], map_location="cpu")["model"],
            strict=False,
        )
        self.darknet.to(device).eval()

        self.kalman_filter = KalmanFilter()

        if "logging_level" in kwargs:
            self.logger.level = kwargs["logging_level"]
            # logging.DEBUG

        # reset member variables to use over a sequence of frame
        self.reset()

    def reset(self):
        # variable manage tracks across a sequence of frames
        self.global_active_tracks = []
        self.global_onhold_tracks = []
        self.global_removed_tracks = []

        self.frame_id = 0

    def input_to_features(self, x) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        return self._input_to_feature_pyramid(x)

    def features_to_output(self, x: Dict):
        """Complete the downstream task from the intermediate deep features"""
        return self._feature_pyramid_to_output(
            x["data"], x["org_input_size"], x["input_size"]
        )

    @torch.no_grad()
    def _input_to_feature_pyramid(self, x):
        """Computes and return feture pyramid all the way from the input"""
        img = x[0]["image"].unsqueeze(0).to(self.device)
        input_size = tuple(img.shape[2:])

        _ = self.darknet(img, self.features_at_splits, is_nn_part1=True)

        return {"data": self.features_at_splits, "input_size": [input_size]}

    @torch.no_grad()
    def get_input_size(self, x):
        """Computes the size of the input image to the network"""
        img = x[0]["image"].unsqueeze(0).to(self.device)
        return tuple(img.shape[2:])

    @torch.no_grad()
    def _feature_pyramid_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs downstream task using the feature pyramid
        """
        pred = self.darknet(None, x, is_nn_part1=False)

        online_targets = self._jde_process(
            pred, (org_img_size["height"], org_img_size["width"]), input_img_size[0]
        )

        online_tlwhs = []
        online_ids = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.model_configs["min_box_area"] and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        return {"tlwhs": online_tlwhs, "ids": online_ids}

    @torch.no_grad()
    def deeper_features_for_accuracy_proxy(self, x: Dict):
        """
        compute accuracy proxy at the deeper layer than NN-Part1
        """
        raise NotImplementedError

        assert x.dim() == 4, "Shape of the input feature tensor must be [N, C, H, W]"
        assert type(tag) == int

        x_deeper = self.darknet(None, {tag: x}, is_nn_part1=False, end_idx=(tag + 1))

        return x_deeper

    def _jde_process(self, pred, org_img_size: tuple, input_img_size: tuple):
        r"""Re-implementation of JDE from Z. Wang, L. Zheng, Y. Liu, and S. Wang:
        : `"Towards Real-Time Multi-Object Tracking"`_,
        The European Conference on Computer Vision (ECCV), 2020

        The implementation refers to
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py>

        Full license statement can be found at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/LICENSE>

        """

        # Active tracks in the current frame
        current_active_tracks = []

        # Re-registred tracks at the current frame
        current_reregistred_tracks = []

        # Missing tracks at the current frame, but still be hold for a while < threshold
        current_onhold_tracks = []

        # removed tracks from the current frame
        current_removed_tracks = []

        assert (
            pred.size(1) == 54264
        ), f"Default number of proposals by JDE must be 54264, but got {pred.size(1)}"

        selected_pred = pred[:, pred[0, :, 4] > self.model_configs["conf_thres"]]
        # only check the objects detected with confidence greater than .5

        if selected_pred.shape[1] > 0:
            # Compute final proposals including bbox and embeddings for detected objects
            res = non_max_suppression(
                selected_pred,
                self.model_configs["conf_thres"],
                self.model_configs["nms_thres"],
            )[0].cpu()

            # Update detection scales

            # Don't understand why round and not use the rounded one?
            _ = scale_coords(input_img_size, res[:, :4], org_img_size)  # .round()
            # Entities in ``res'' are following (x1, y1, x2, y2, object_conf, class_score, class_pred or embeddings)'''

            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4].numpy()), tlbrs[4], f.numpy(), 30)
                for (tlbrs, f) in zip(res[:, :5], res[:, 6:])
            ]
        else:
            detections = []

        # remove any detected tracks with zero height
        n_detected_objs = len(detections)
        detections = [d for d in detections if d._tlwh[3] != 0]

        if len(detections) < n_detected_objs:
            self.logger.warning(
                f"Original number of detected objetcs is {n_detected_objs}, but got reduced to {len(detections)} by discarding zero height entities\n"
            )

        # Step 1: Sort out unactive tracklets
        _inactive_tracks = []
        _active_tracks = []
        for track in self.global_active_tracks:
            if not track.is_activated:
                # Sort out tracks inactive in the current frame
                _inactive_tracks.append(track)
            else:
                # Active tracks are listed
                _active_tracks.append(track)

        # Step 2: First association, with embeddings

        # Union tracks of _active_tracks and global_onhold_stracks
        track_candidates_pool = joint_stracks(_active_tracks, self.global_onhold_tracks)

        # Predict the current location with KF
        STrack.multi_predict(track_candidates_pool, self.kalman_filter)

        # calculate distances between the detections with the tracks in track candidate pool
        dists = matching.embedding_distance(track_candidates_pool, detections)
        dists = matching.fuse_motion(
            self.kalman_filter, dists, track_candidates_pool, detections
        )

        # The matches is the array for tracks whose distance is shorter then threshold
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = track_candidates_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is in active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                current_active_tracks.append(track)
            else:
                # Reactive and register track in active but detected again in the current frame
                # to the current reregistered track list
                track.re_activate(det, self.frame_id, new_id=False)
                current_reregistred_tracks.append(track)

        # None of the steps below happens if there are no undetected tracks.
        # Step 3: Second association, with IOU'''

        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = (
            []
        )  # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if track_candidates_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(track_candidates_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                current_active_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                current_reregistred_tracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                current_onhold_tracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are marked lost and are added to the list of onhold tracks

        """Deal with inactive tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(_inactive_tracks, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            _inactive_tracks[itracked].update(detections[idet], self.frame_id)
            current_active_tracks.append(_inactive_tracks[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = _inactive_tracks[it]
            track.mark_removed()
            current_removed_tracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        # Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.model_configs["conf_thres"]:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            current_active_tracks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.global_onhold_tracks:
            if self.frame_id - track.end_frame > self.max_time_on_hold:
                track.mark_removed()
                current_removed_tracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.global_active_tracks = [
            t for t in self.global_active_tracks if t.state == TrackState.Tracked
        ]
        self.global_active_tracks = joint_stracks(
            self.global_active_tracks, current_active_tracks
        )
        self.global_active_tracks = joint_stracks(
            self.global_active_tracks, current_reregistred_tracks
        )
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.global_onhold_tracks = sub_stracks(
            self.global_onhold_tracks, self.global_active_tracks
        )
        self.global_onhold_tracks.extend(current_onhold_tracks)
        self.global_onhold_tracks = sub_stracks(
            self.global_onhold_tracks, self.global_removed_tracks
        )
        self.global_removed_tracks.extend(current_removed_tracks)
        self.global_active_tracks, self.global_onhold_tracks = remove_duplicate_stracks(
            self.global_active_tracks, self.global_onhold_tracks
        )

        # get scores of lost tracks
        output_stracks = [
            track for track in self.global_active_tracks if track.is_activated
        ]

        self.frame_id += 1

        self.logger.debug(f"===========Frame {self.frame_id}==========")
        self.logger.debug(
            "Activated    : {}".format(
                [track.track_id for track in current_active_tracks]
            )
        )
        self.logger.debug(
            "Re-registered: {}".format(
                [track.track_id for track in current_reregistred_tracks]
            )
        )
        self.logger.debug(
            "On hold      : {}".format(
                [track.track_id for track in current_onhold_tracks]
            )
        )
        self.logger.debug(
            "Removed      : {}".format(
                [track.track_id for track in current_removed_tracks]
            )
        )

        return output_stracks

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        # return self.model(x)
        raise NotImplementedError

    def reshape_feature_pyramid_to_frame(
        self, features: Dict, packing_all_in_one=False
    ):
        """rehape the feature pyramid to a frame"""

        # tensors ordered by width dimention
        sorted_feature_by_width = dict(
            sorted(features.items(), key=lambda x: x[1].size()[2], reverse=True)
        )
        sorted_keys = sorted_feature_by_width.keys()

        nbframes, C, H, W = sorted_feature_by_width[list(sorted_keys)[0]].size()
        _, fixedW = compute_frame_resolution(C, H, W)

        packed_frames = {}
        feature_size = {}
        subframe_heights = {}
        subframe_widths = {}

        packed_frame_list = []
        for n in range(nbframes):
            for key, tensor in sorted_feature_by_width.items():
                single_tensor = tensor[n : n + 1, ::]
                N, C, H, W = single_tensor.size()

                assert N == 1, f"the batch size shall be one, but got {N}"

                if n == 0:
                    feature_size.update({key: single_tensor.size()})

                    frmH, frmW = compute_frame_resolution(C, H, W)

                    rescale = fixedW // frmW if packing_all_in_one else 1

                    new_frmH = frmH // rescale
                    new_frmW = frmW * rescale

                    subframe_heights.update({key: new_frmH})
                    subframe_widths.update({key: new_frmW})

                tile = tensor_to_tiled(
                    single_tensor, (subframe_heights[key], subframe_widths[key])
                )

                packed_frames.update({key: tile})

            if packing_all_in_one:
                packed_frame = torch.cat(list(packed_frames.values()), dim=0)
                packed_frame_list.append(packed_frame)

        packed_frames = torch.stack(packed_frame_list)

        return packed_frames, feature_size, subframe_heights

    def reshape_frame_to_feature_pyramid(
        self, x, tensor_shape: Dict, subframe_height: Dict, packing_all_in_one=False
    ):
        """reshape a frame of channels into the feature pyramid"""

        assert isinstance(x, (Tensor, Dict))

        top_y = 0
        tiled_frames = {}
        if packing_all_in_one:
            for key, height in subframe_height.items():
                tiled_frames.update({key: x[:, top_y : top_y + height, :]})
                top_y = top_y + height
        else:
            raise NotImplementedError
            assert isinstance(x, Dict)
            tiled_frames = x

        feature_tensor = {}
        for key, frames in tiled_frames.items():
            _, numChs, chH, chW = tensor_shape[key]

            tensors = []
            for frame in frames:
                tensor = tiled_to_tensor(frame, (chH, chW)).to(self.device)
                tensors.append(tensor)
            tensors = torch.cat(tensors, dim=0)
            assert tensors.size(1) == numChs

            feature_tensor.update({key: tensors})

        return feature_tensor
