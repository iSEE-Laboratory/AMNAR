import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
try:
    from .datasets import register_dataset
    from .data_utils import truncate_feats
except ImportError:
    from datasets import register_dataset
    from data_utils import truncate_feats
import pickle
import logging

@register_dataset("ATA")
class ATADataset(Dataset):
    def __init__(
        self,
        is_training,        # 是否为训练模式
        split,              # 数据集划分
        default_fps,        # 默认帧率
        max_seq_len,        # 训练时的最大序列长度
        trunc_thresh,       # 截断阈值
        crop_ratio,         # 随机裁剪比例
        **kwargs
    ):
        # 设置根目录和特征目录
        root_dir = '/home/weijin/source/MistakeDetection/FAFP/ata'
        self.feat_path = os.path.join(root_dir, 'ata_data/I3D_Features/i3d_1024d')
        self.results_inference_path = os.path.join(root_dir, 'results_inference')
        
        # 初始化基本属性
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        
        # 读取数据列表
        with open(os.path.join(root_dir, 'ata_data/Temporal_Annotations', f'split.{split}'), 'r') as fp:
            lines = fp.readlines()
            self.data_list = [line.strip('\n') for line in lines]
        

        action_to_idx_path = os.path.join(self.results_inference_path, 'action_to_idx.pkl')
        with open(action_to_idx_path, 'rb') as fp:
            self.action_to_idx = pickle.load(fp)

        idx_to_action_path = os.path.join(self.results_inference_path, 'idx_to_action.pkl')
        with open(idx_to_action_path, 'rb') as fp:
            self.idx_to_action = pickle.load(fp)


        if split == 'train':
            results_dict_path = os.path.join(self.results_inference_path, 'results_dict.pkl')
            with open(results_dict_path, 'rb') as fp:
                self.train_results_dict = pickle.load(fp)

            self.data_list = []
            for key, value in self.train_results_dict.items():
                value['video_id'] = key
                self.data_list.append(value)
        elif split == 'test':
            results_dict_path = os.path.join(self.results_inference_path, 'results_dict.pkl')
            with open(results_dict_path, 'rb') as fp:
                self.train_results_dict = pickle.load(fp)

            trainset_samples = []
            for key, value in self.train_results_dict.items():
                trainset_samples.append(key)

            print(len(trainset_samples))
            print(trainset_samples)
            gt_dict_path = os.path.join(self.results_inference_path, 'gt_dict.pkl')
            with open(gt_dict_path, 'rb') as fp:
                self.gt_dict = pickle.load(fp)

            self.data_list = []
            for key, value in self.gt_dict.items():
                if key in trainset_samples:
                    continue
                value['video_id'] = key
                self.data_list.append(value)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        sample_data = self.data_list[idx]

        video_id = sample_data['video_id']

        feat_path = os.path.join(self.feat_path, f'{video_id}.npy')
        feats = np.load(feat_path) # Size: (T, 2048)

        segments = sample_data['segments']
        action_labels = sample_data['action_labels']
        
        # 构建数据字典
        data_dict = {
            'feats': torch.from_numpy(feats).permute(1, 0).float(),
            'segments': torch.from_numpy(time_stamps).float(),
            'labels': torch.from_numpy(action_labels).long(),
            'video_id': str(video_id),
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
            'action_id_to_str': self.idx_to_action,
        }
        
        # 训练时截断特征
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)
            
        return data_dict


if __name__ == '__main__':
    dataset = ATADataset(is_training=True, split='test', default_fps=25, max_seq_len=100, trunc_thresh=0.5, crop_ratio=0.5)
    print(len(dataset))
    for i in range(len(dataset)):
        sample_data = dataset[i]
        print(sample_data['video_id'])
        assert False
