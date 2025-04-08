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

@register_dataset("HoloAssist")
class HoloAssistDataset(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        split,              # split, a tuple/list allowing concat of subsets
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        task,               # task name
        default_fps = 10,
        segment_type = 'n', # n or v or vn
        **kwargs
    ):
        root_dir = '/home/weijin/source/MistakeDetection/FAFP/HoloAssist'
        self.feat_root_dir = '/home/weijin/source/MistakeDetection/FAFP/HoloAssist/feats/I3D'
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.task = task
        self.segment_type = segment_type

        # 读取数据列表
        with open(os.path.join(root_dir, 'data-task-splits', self.split, f'{task}.txt'), 'r') as fp:
            lines = fp.readlines()
            self.data_list = [line.strip('\n') for line in lines]

        # 读取标注信息
        with open(os.path.join(root_dir, 'data-annotation-trainval-v1.json'), 'r') as fp:
            all_annot = json.load(fp)

        annots = []
        all_labels = []
        for sample_data in all_annot:
            if sample_data['video_name'] in self.data_list:
                feat_path = os.path.join(self.feat_root_dir, sample_data['video_name']+'.npy')
                if not os.path.exists(feat_path):
                    logging.info(f'{feat_path} not exists')
                    continue
                sample_data['feat_path'] = feat_path
                
                events = sample_data['events']
                frame_length = len(np.load(feat_path))
                
                sample_segments = []
                sample_labels = []
                sample_labels_error = []
                for event in events:
                    if event['label'] == 'Fine grained action':
                        v = event['attributes']['Verb']       
                        n = event['attributes']['Noun']
                        if segment_type == 'n':
                            label_str = n
                        elif segment_type == 'v':
                            label_str = v
                        elif segment_type == 'vn':
                            label_str = f"{v}-{n}"  # 使用连字符连接动词和名词

                        start = int(event['start'] * self.default_fps)
                        end = int(event['end'] * self.default_fps)
                        if end > frame_length - 1:
                            end = frame_length - 1
                        if start > end:
                            print(f'start > end: {start} {end} in {sample_data["video_name"]}')
                            continue
                        
                        all_labels.append(label_str)
                        segment = [start, end]
                        sample_segments.append(segment)
                        sample_labels.append(label_str)

                        correctness = event['attributes']['Action Correctness']
                        is_wrong = 'wrong' in correctness.lower()
                        label_wrong = -1 if is_wrong else 0
                        sample_labels_error.append(label_wrong)

                merged_segments = []
                merged_labels = []
                merged_labels_error = []

                if sample_segments:  # 确保有片段需要处理
                    current_segment = sample_segments[0]
                    current_label = sample_labels[0]
                    current_error = sample_labels_error[0]
                    
                    for i in range(1, len(sample_segments)):
                        if (sample_segments[i][0] - current_segment[1] <= 1 and 
                            sample_labels[i] == current_label):
                            # 更新当前片段的结束时间
                            current_segment[1] = sample_segments[i][1]
                            # 如果任一片段有错误，则合并后的片段标记为错误
                            current_error = -1 if current_error == -1 or sample_labels_error[i] == -1 else 0
                        else:
                            # 保存当前片段并开始新片段
                            merged_segments.append(current_segment)
                            merged_labels.append(current_label)
                            merged_labels_error.append(current_error)
                            current_segment = sample_segments[i]
                            current_label = sample_labels[i]
                            current_error = sample_labels_error[i]
                    
                    # 添加最后一个片段
                    merged_segments.append(current_segment)
                    merged_labels.append(current_label)
                    merged_labels_error.append(current_error)
                    
                    # 更新原始列表
                    sample_segments = merged_segments
                    sample_labels = merged_labels
                    sample_labels_error = merged_labels_error                        

                bg_start = 0
                bg_segments = []
                bg_labels = []
                bg_labels_error = []
                for segment in sample_segments:
                    if bg_start < segment[0]:
                        bg_segments.append([bg_start, segment[0]-1])
                        bg_labels.append('bg')
                        bg_labels_error.append(0)
                    bg_start = segment[1] + 1

                if bg_start < frame_length:
                    bg_segments.append([bg_start, frame_length-1])
                    bg_labels.append('bg')
                    bg_labels_error.append(0)

                sample_segments.extend(bg_segments)
                sample_labels.extend(bg_labels)
                sample_labels_error.extend(bg_labels_error)
                # 按segment[0]排序,获取排序后的索引
                sorted_indexes = sorted(range(len(sample_segments)), key=lambda x: sample_segments[x][0])

                # 根据索引对sample_segments和sample_labels重排序
                sample_segments = [sample_segments[idx] for idx in sorted_indexes]
                sample_labels = [sample_labels[idx] for idx in sorted_indexes]
                sample_labels_error = [sample_labels_error[idx] for idx in sorted_indexes]

                # 将片段和标签信息添加到sample_data中 
                sample_data['segments'] = sample_segments
                sample_data['labels_str'] = sample_labels
                sample_data['labels_error'] = sample_labels_error

                annots.append(sample_data)

        self.annots = annots
        # 构造idx_to_label/label_to_idx
        label_dict_path = f'/home/weijin/source/MistakeDetection/FAFP/dataset_buffer/holoassist_label_dict_{task}_{segment_type}.pkl'
        if os.path.exists(label_dict_path):
            with open(label_dict_path, 'rb') as f:
                label_dict = pickle.load(f)
            self.idx_to_label = label_dict['idx_to_label']
            self.label_to_idx = label_dict['label_to_idx']
        else:
            self.idx_to_label = ['bg'] + list(set(all_labels))  # 在列表开头添加 'bg'
            self.label_to_idx = {label: idx for idx, label in enumerate(self.idx_to_label)}
            with open(label_dict_path, 'wb') as f:
                pickle.dump({'idx_to_label': self.idx_to_label, 'label_to_idx': self.label_to_idx}, f)

        self.num_classes = len(self.idx_to_label)
        print(f'num_classes: {self.num_classes}')

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        annot = self.annots[idx]

        video_name = annot['video_name']
        events = annot['events']
        feat_path = annot['feat_path']
        feats = np.load(feat_path) # Shape: (T, 2048)

        segments = annot['segments']
        labels_str = annot['labels_str']
        labels = [self.label_to_idx.get(label, 0) for label in labels_str]
        labels_error = annot['labels_error']

        # to tensor
        segments = torch.tensor(segments).float()
        labels = torch.tensor(labels).long()
        labels_error = torch.tensor(labels_error).long()

        data_dict = {
            'video_id': video_name,
            'feats': torch.from_numpy(feats).permute(1, 0).float(),
            'segments': segments,
            'labels': labels,
            'labels_error': labels_error,
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
            'action_id_to_str': self.idx_to_label,
        }
        # print('video_id: ', data_dict['video_id'])
        # print('labels: ', data_dict['labels'])
        # print('labels_error: ', data_dict['labels_error'])
        # print('segments: ', data_dict['segments'])
        # print('feats: ', data_dict['feats'].shape)
        # print('fps: ', data_dict['fps'])
        # print('duration: ', data_dict['duration'])
        # print('action_id_to_str: ', data_dict['action_id_to_str'])
        # assert False

        # truncate the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict
    

    def save_processed_sequences(self, save_path):
        """处理并保存所有动作序列
        1. 移除所有action_idx 0
        2. 添加起始节点(0)和结束节点(max_label + 1)
        """
        # 找到最大的action_idx作为结束节点的标号
        max_label = max(self.label_to_idx.values())
        end_node = max_label + 1
        
        # 存储所有处理后的序列
        processed_sequences = []
        
        # 遍历数据集处理序列
        for annot in self.annots:
            # 获取标签序列并移除所有的背景类(action_idx 0)
            labels = [self.label_to_idx.get(label, 0) for label in annot['labels_str']]
            labels = [l for l in labels if l != 0]
            
            if not labels:  # 如果序列为空，跳过
                continue
            
            # 合并连续的相同动作
            merged_labels = []
            current_label = labels[0]
            for label in labels[1:]:
                if label != current_label:
                    merged_labels.append(current_label)
                    current_label = label
            merged_labels.append(current_label)  # 添加最后一个动作
                
            # 添加起始节点和结束节点
            sequence = [0] + merged_labels + [end_node]
            processed_sequences.append(sequence)
        
        # 保存结果
        result = {
            'task': self.task,
            'segment_type': self.segment_type,
            'max_label': max_label,
            'end_node': end_node,
            'sequences': processed_sequences,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': {str(v): k for k, v in self.label_to_idx.items()}  # 转换key为字符串，因为json不支持整数key
        }
        
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存为json文件
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return processed_sequences

"""
num_classes:
n:

atv 17
belt 7
circuitbreaker 13
coffee 38
computer 19
dslr 20
gladom_assemble 20
gladom_disassemble 18
gopro 24
knarrevik_assemble 12
knarrevik_disassemble 8
marius_assemble 16
marius_disassemble 15
navvis 48
nespresso 29
printer_big 21
printer_small 18
rashult_assemble 18
rashult_disassemble 20
switch 21

v:
atv 24
belt 20
circuitbreaker 25
coffee 30
computer 29
dslr 30
gladom_assemble 29
gladom_disassemble 21
gopro 37
knarrevik_assemble 27
knarrevik_disassemble 21
marius_assemble 24
marius_disassemble 23
navvis 34
nespresso 31
printer_big 24
printer_small 33
rashult_assemble 27
rashult_disassemble 26
switch 33
"""





if __name__ == '__main__':
    # 为每个任务处理序列
    tasks = ['coffee', 'atv', 'belt', 'circuitbreaker', 'computer', 
             'dslr', 'gladom_assemble', 'gladom_disassemble', 'gopro',
             'knarrevik_assemble', 'knarrevik_disassemble', 'marius_assemble',
             'marius_disassemble', 'navvis', 'nespresso', 'printer_big',
             'printer_small', 'rashult_assemble', 'rashult_disassemble', 'switch']
    
    for task in tasks:
        for segment_type in ['n', 'v', 'vn']:  # 添加 'vn' 类型
            dataset = HoloAssistDataset(
                is_training=True,
                split='train',
                max_seq_len=2304,
                trunc_thresh=0.5,
                crop_ratio=[0.9, 1.0],
                task=task,
                segment_type=segment_type
            )
            
            os.makedirs('./holoassist_sequences_buffer', exist_ok=True)
            save_path = f'./holoassist_sequences_buffer/sequences_{task}_{segment_type}.json'
            sequences = dataset.save_processed_sequences(save_path)
            print(f"已保存 {task} ({segment_type}) 的序列，共 {len(sequences)} 条")
