from typing import List, Union
import pandas
import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset
from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON, COCO_SKELETON


class OpenPoseDataset(Dataset):
    def __init__(self, data_dir, set_filepath, points: Union[BODY_25_SKELETON, COCO_SKELETON] = BODY_25_SKELETON, transform=None) -> None:
        self.data_dir = data_dir

        self.clips = pandas.read_csv(set_filepath)
        self.clips.set_index(['video', 'id', 'clip'], inplace=True)
        self.clips.sort_index(inplace=True)

        self.indices = pandas.MultiIndex.from_frame(
            self.clips.index.to_frame(index=False).drop_duplicates())

        self.transform = transform

        self.points = points

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Returns a single clip.

        :param idx: Clip index
        :type idx: int
        """
        clips_idx = self.indices[idx]
        pedestrian_info = self.clips.loc[clips_idx].reset_index().sort_values('frame')

        (video_id, pedestrian_id, clip_id) = clips_idx
        start_frame = pedestrian_info.iloc[0]['frame']
        stop_frame = pedestrian_info.iloc[-1]['frame'] + 1
        frames = []
        bboxes = []

        for i, f in enumerate(range(start_frame, stop_frame, 1)):
            gt_bbox = pedestrian_info.iloc[i][[
                'x1', 'y1', 'x2', 'y2']].to_numpy().reshape((2, 2)).astype(np.float32)
            bboxes.append(torch.tensor(gt_bbox))
            with open(os.path.join(self.data_dir, video_id, '{:s}_{:0>12d}_keypoints.json'.format(
                video_id,
                f
            ))) as jp:
                people = json.load(jp)['people']
                if not len(people):
                    # OpenPose didn't detect anything in this frame - append empty array
                    frames.append(np.zeros((len(self.points), 3)))
                else:
                    # select the pose with biggest IOU with base bounding box
                    candidates = [np.array(p['pose_keypoints_2d']).reshape(
                        (-1, 3)) for p in people]
                    frames.append(self.__select_best_candidate(candidates, gt_bbox))

        torch_frames = torch.tensor(frames, dtype=torch.float32)

        if self.transform is not None:
            torch_frames = self.transform(torch_frames)

        return (torch_frames, {}, {
            'age': pedestrian_info.iloc[0]['age'],
            'gender': pedestrian_info.iloc[0]['gender'],
            'video_id': video_id,
            'pedestrian_id': pedestrian_id,
            'clip_id': clip_id,
            'start_frame': start_frame,
            'end_frame': stop_frame,
            'bboxes': torch.stack(bboxes, dim=0)
        })

    def __select_best_candidate(self, candidates: List[np.ndarray], gt_bbox: np.ndarray, near_zero: float = 1e-5) -> np.ndarray:
        """
        Selects the pose with the biggest overlap with ground truth bounding box.
        If the IOU is smaller than `near_zero` value, it returns empty array.

        :param candidates: [description]
        :type candidates: List[np.ndarray]
        :param gt_bbox: [description]
        :type gt_bbox: np.ndarray
        :param near_zero: [description], defaults to 1e-5
        :type near_zero: float, optional
        :return: Best pose candidate for specified bounding box.
        :rtype: np.ndarray
        """
        candidates_bbox = np.array([
            np.array([
                c[:, 0:2].min(axis=0),
                c[:, 0:2].max(axis=0)
            ])
            for c in candidates
        ])

        gt_min = gt_bbox.min(axis=0)
        candidates_min = candidates_bbox.min(axis=1)

        gt_max = gt_bbox.max(axis=0)
        candidates_max = candidates_bbox.max(axis=1)

        intersection_min = np.maximum(gt_min, candidates_min)
        intersection_max = np.minimum(gt_max, candidates_max)

        intersection_area = (intersection_max -
                             intersection_min + 1).prod(axis=1)
        gt_area = (gt_max - gt_min + 1).prod(axis=0)
        candidates_area = (candidates_max - candidates_min + 1).prod(axis=1)

        iou = intersection_area / \
            (gt_area + candidates_area - intersection_area)

        best_iou_idx = np.argmax(iou)

        if iou[best_iou_idx] < near_zero:
            return np.zeros((len(self.points), 3))

        return candidates[best_iou_idx]
