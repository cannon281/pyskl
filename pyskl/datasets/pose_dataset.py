# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import os.path as osp

from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS
import random
import numpy as np
import math
from collections import Counter

@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        use_3d_keypoints (bool): Whether to use 3D keypoints instead of 2D. Default: False.
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 preprocess_type="",
                 use_3d_keypoints=False,
                 **kwargs):
        modality = 'Pose'
        self.split = split
        self.use_3d_keypoints = use_3d_keypoints

        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, memcached=memcached, mc_cfg=mc_cfg, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None and isinstance(self.valid_ratio, float) and self.valid_ratio > 0:
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                assert 'box_score' in item, 'if valid_ratio is a positive number, item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)
            if self.memcached:
                item['key'] = item['frame_dir']

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')
        
        if preprocess_type == "sequential":
            logger.info(f"Using sequential preprocessing with 3D keypoints: {self.use_3d_keypoints}")
            self.clip_length = 144
            self.skip_length = 20
            self.model_clip_length = 48
            self.preprocess_mv(preprocess_type)
        elif preprocess_type == "skip":
            logger.info(f"Using skip preprocessing with 3D keypoints: {self.use_3d_keypoints}")
            self.clip_length = 164
            self.skip_length = 20
            self.model_clip_length = 48
            self.preprocess_mv(preprocess_type)

    def preprocess_mv(self, preprocess_type):
        """Preprocess long videos with moving window sampling to form small clips."""
        new_video_infos = []
        total_valid_indices = 0
        
        logger = get_root_logger()
        logger.info(f"Preprocess type: {preprocess_type}")
        logger.info(f"Using 3D keypoints: {self.use_3d_keypoints}")
        
        for sample in self.video_infos:            
            total_frames = sample['total_frames']
            labels = sample['labels']
            current_index = 0
            
            while True:
                if current_index >= total_frames:
                    break
                
                end_index = min(current_index + self.clip_length, total_frames)
                if end_index == total_frames:
                    break
                    
                frames_to_sample = end_index - current_index
                if frames_to_sample >= self.clip_length:
                    # Sample keypoints
                    new_keypoints = sample['keypoint'][:, current_index:end_index, :, :]
                    
                    if self.use_3d_keypoints:
                        # For 3D keypoints, we don't need keypoint_score
                        # Create dummy scores for compatibility (won't be used)
                        new_keypoints_score = np.ones((new_keypoints.shape[0], new_keypoints.shape[1], new_keypoints.shape[2]), dtype='float16')
                    else:
                        # For 2D keypoints, use the actual scores
                        new_keypoints_score = sample['keypoint_score'][:, current_index:end_index, :]
                        new_keypoints_score = np.nan_to_num(new_keypoints_score)
                        new_keypoints_score = np.clip(new_keypoints_score, 0, 1)
                    
                    new_labels = sample['labels'][current_index:end_index]
                    
                    if Counter(new_labels).most_common(1)[0][0] == 0:
                        current_index = current_index + self.skip_length
                        continue  # Skip this clip
                    
                    non_skip_usable = []
                    non_skip_usable_label = []

                    for i in range(self.skip_length):
                        if preprocess_type == "sequential":
                            label_from_center = self.get_majority_center_label(new_labels[i:self.model_clip_length+i], center_frames=5)
                            keypoints_abs_sum = np.sum(new_keypoints[:, i:self.model_clip_length+i, :, :] == 0)
                        else:
                            label_from_center = self.get_majority_center_label(new_labels[i:(self.model_clip_length*3)+i][::3], center_frames=5)
                            keypoints_abs_sum = np.sum(new_keypoints[:, i:(self.model_clip_length*3)+i, :, :][:, ::3, :, :] == 0)
                    
                        # Adjust threshold for 3D keypoints (3D has 3 coords vs 2 for 2D + confidence)
                        threshold = 255 if self.use_3d_keypoints else 170
                        if label_from_center != 0 and keypoints_abs_sum <= threshold:
                            non_skip_usable.append(i)
                            non_skip_usable_label.append(label_from_center)
                    
                    new_keypoints_score = new_keypoints_score.astype('float16')

                    if len(non_skip_usable) > 0:
                        total_valid_indices = total_valid_indices + len(non_skip_usable)
                        if self.split == "sub_test":
                            for idx, x in enumerate(non_skip_usable):  # separate each usable index as a sample for testing
                                new_sample = {
                                    'frame_dir': sample['frame_dir'],
                                    'labels': new_labels,
                                    'img_shape': sample['img_shape'],
                                    'original_shape': sample['img_shape'],
                                    'total_frames': frames_to_sample,
                                    'usable_indices': np.array([non_skip_usable[idx]]),
                                    'label': (int(non_skip_usable_label[idx])-1),
                                    'usable_label': np.array([non_skip_usable_label[idx]]),
                                    'keypoint': new_keypoints
                                }
                                # Only add keypoint_score for 2D keypoints
                                if not self.use_3d_keypoints:
                                    new_sample['keypoint_score'] = new_keypoints_score
                                new_video_infos.append(new_sample)
                        else:
                            new_sample = {
                                'frame_dir': sample['frame_dir'],
                                'labels': new_labels,
                                'img_shape': sample['img_shape'],
                                'original_shape': sample['img_shape'],
                                'total_frames': frames_to_sample,
                                'usable_indices': np.array(non_skip_usable),
                                'usable_label': np.array(non_skip_usable_label),
                                'keypoint': new_keypoints
                            }
                            # Only add keypoint_score for 2D keypoints
                            if not self.use_3d_keypoints:
                                new_sample['keypoint_score'] = new_keypoints_score
                            new_video_infos.append(new_sample)               
                    current_index = current_index + self.skip_length
                else:
                    break
                    
        self.video_infos = new_video_infos
        logger.info(f"Dataset sample length: {len(self.video_infos)}")
        logger.info(f"Total valid indices: {total_valid_indices}")

    def get_majority_center_label(self, labels, center_frames):
        # Calculate the start and end indices for the center frames
        start = len(labels) // 2 - center_frames // 2
        end = start + center_frames

        # Extract the center labels
        center_labels = labels[start:end]

        # Count the frequency of each label in the center frames
        label_counts = Counter(center_labels)

        # Return the most common label
        return label_counts.most_common(1)[0][0]
        
    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        logger = get_root_logger()
        
        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
            
            # Handle 3D keypoints if specified
            if self.use_3d_keypoints:
                if 'keypoint_3d' in item:
                    # Replace 2D keypoints with 3D keypoints
                    kp_3d = item['keypoint_3d']
                    
                    # Convert NaN to zeros to prevent NaN losses
                    # The existing filtering logic will handle windows with too many zeros
                    if np.any(np.isnan(kp_3d)):
                        logger.debug(f"Converting NaN to zeros in 3D keypoints for {item.get('frame_dir', 'unknown')}")
                        kp_3d = np.nan_to_num(kp_3d, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    item['keypoint'] = kp_3d
                    # Remove keypoint_score to avoid issues with pipelines that expect 2D
                    item.pop('keypoint_score', None)
                    
                    logger.debug(f"Using 3D keypoints for {item.get('frame_dir', 'unknown')}: "
                               f"keypoint shape={item['keypoint'].shape}")
                else:
                    logger.warning(f"3D keypoints requested but 'keypoint_3d' not found in {item.get('frame_dir', 'unknown')}")
        
        return data