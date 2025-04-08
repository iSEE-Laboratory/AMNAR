import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import logging
try:
    from .datasets import register_dataset
    from .data_utils import truncate_feats
except ImportError:
    from datasets import register_dataset
    from data_utils import truncate_feats
import pickle
from collections import defaultdict
import heapq

@register_dataset("CaptainCook4D")
class CaptainCook4DDataset(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        split,              # split, a tuple/list allowing concat of subsets
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        task,               # task name
        default_fps = 10,
        features_subdir = 'I3D',
        **kwargs
    ):
        root_dir = '/home/weijin/source/MistakeDetection/FAFP/CaptainCook4D'
        self.feat_root_dir = os.path.join(root_dir, 'feats', features_subdir)
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.task = task


        if self.split == 'train':
            annotation_path = os.path.join(root_dir, 'CC4D_process', 'non_error_samples_processed.json')
        elif self.split == 'test':
            annotation_path = os.path.join(root_dir, 'CC4D_process', 'error_samples_processed.json')
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        with open(annotation_path, 'r') as f:
            self.annotation_data = json.load(f)[self.task]


        self.annots = []
        for video_id, video_data in self.annotation_data.items():
            # feat
            if self.task != '0':
                feat_path = os.path.join(self.feat_root_dir, f'{video_id}_360p.npy')
            else:
                temp_video_id = video_id.split('_', 1)[1]
                feat_path = os.path.join(self.feat_root_dir, f'{temp_video_id}_360p.npy')
            
            # 检查特征文件是否存在
            if not os.path.exists(feat_path):
                continue
        
            # segments
            segments = video_data['segments']

            # labels
            labels = video_data['labels']

            # labels_error
            labels_error = video_data['labels_error']

            # descriptions
            descriptions = video_data['descriptions']

            self.annots.append({
                'video_id': video_id,
                'feat_path': feat_path,
                'segments': segments,
                'labels': labels,
                'labels_error': labels_error,
                'descriptions': descriptions
            })

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        annot = self.annots[idx]

        video_id = annot['video_id']
        feat_path = annot['feat_path']
        feats = np.load(feat_path) # Shape: (T, 2048)

        segments = annot['segments']
        labels = annot['labels']
        labels_error = annot['labels_error']
        descriptions = annot['descriptions']
        # to tensor
        segments = torch.tensor(segments).float()
        labels = torch.tensor(labels).long()
        labels_error = torch.tensor(labels_error).long()

        data_dict = {
            'video_id': video_id,
            'feats': torch.from_numpy(feats).transpose(0, 1).float(),
            'segments': segments,
            'labels': labels,
            'labels_error': labels_error,
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
        }



        # truncate the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict



if __name__ == '__main__':
    # 为每个任务处理序列
    task='1'
    dataset = CaptainCook4DDataset(
        is_training=True,
        split='train',
        max_seq_len=2304,
        trunc_thresh=0.5,
        crop_ratio=[0.9, 1.0],
        task=task,
    )

    print(len(dataset))
    for sample in dataset:
        print(sample)
        assert False
