import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from .datasets import register_dataset
from .data_utils import truncate_feats, generate_node_connected, truncate_online_data, extract_text_features
import pickle

@register_dataset("EgoPER")
class EgoPERdataset(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        split,              # split, a tuple/list allowing concat of subsets
        default_fps,        # default fps
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        height,             # height of the frame (default: 720)
        width,              # width of the frame (default: 1280)
        num_classes,        # num of action classes
        background_ratio,   # ratio of sampled background
        num_node,           # num of nodes in a graph
        use_gcn,            # if using AOD
        task,               # task name
        online_test,        # if in online testing mode
        ckpt_folder = None, # checkpoint folder
        features_subdir = 'features_10fps_new',
        **kwargs
    ):
        root_dir = '/home/weijin/source/MistakeDetection/EgoPER_official/EgoPER'
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.background_ratio = background_ratio
        self.use_gcn = use_gcn
        self.bg_idx = 0
        self.annotations = {}
        self.online_test = online_test
        self.feat_path = os.path.join(root_dir, task, features_subdir)

        
        with open(os.path.join(root_dir, task, self.split+'.txt'), 'r') as fp:
            lines = fp.readlines()
            self.data_list = [line.strip('\n') for line in lines]
        with open(os.path.join(root_dir, 'annotation.json'), 'r') as fp:
            all_annot = json.load(fp)
        

        self.data_list_online_test = []

        # 构造dataset & 构造task对应的action order
        annot = all_annot[task]
        
        task_action_order = {}
        error_descriptions = ["A new action"]
        self.error_descriptions_to_feat = None
        self.action_id_to_str = {-1: "A new action"}

        for i in range(len(annot['segments'])):
            video_id = annot['segments'][i]['video_id']
            if video_id in self.data_list:
                actions = [int(action) for action in annot['segments'][i]['labels']['action']]
                action_types = [int(action_type) for action_type in annot['segments'][i]['labels']['action_type']]
                sample_error_descriptions = annot['segments'][i]['labels']['error_description']

                if not is_training and ckpt_folder is not None:
                    with open(os.path.join(ckpt_folder, f'action_id_to_str_{task}.json'), 'r') as fp:
                        self.action_id_to_str = json.load(fp)
                        self.action_id_to_str = {int(k): v for k, v in self.action_id_to_str.items()}
                    with open(os.path.join(ckpt_folder, f'error_descriptions_to_feat_{task}.pkl'), 'rb') as fp:
                        self.error_descriptions_to_feat = pickle.load(fp)
                    for idx, (action, error_description) in enumerate(zip(actions, sample_error_descriptions)):
                        if self.action_id_to_str[action] != error_description:
                            sample_error_descriptions[idx] = self.action_id_to_str[action]
                else:
                    # 保存error_description 后面特征映射 & 保留 action id to str 映射
                    for action, error_description in zip(actions, sample_error_descriptions):
                        if error_description not in error_descriptions:
                            error_descriptions.append(error_description)
                        if action not in self.action_id_to_str:
                            self.action_id_to_str[action] = error_description
                        else:
                            assert self.action_id_to_str[action] == error_description
                
                num_seg = len(actions)
                for j in range(num_seg):
                    self.data_list_online_test.append((video_id, j))
                self.annotations[video_id] = [np.array(annot['segments'][i]['labels']['time_stamp']) * self.default_fps, 
                                            np.array(actions),
                                            np.array(action_types),
                                            sample_error_descriptions]
                task_action_order[video_id] = actions


        task_action_order_path = 'trainset_action_order.json'
        if not os.path.exists(task_action_order_path):
            trainset_action_order = {task: task_action_order}
        else:
            try:
                with open(task_action_order_path, 'r') as fp:
                    trainset_action_order = json.load(fp)
            except:
                trainset_action_order = {}
            trainset_action_order[task] = task_action_order

        with open(task_action_order_path, 'w') as fp:
            json.dump(trainset_action_order, fp, indent=4)            


        if is_training:
            self.error_descriptions_to_feat = extract_text_features(error_descriptions)



        # SAVE action_id_to_str AND error_descriptions_to_feat
        if is_training and ckpt_folder is not None:
            with open(os.path.join(ckpt_folder, f'action_id_to_str_{task}.json'), 'w') as fp:
                json.dump(self.action_id_to_str, fp, indent=4)
            with open(os.path.join(ckpt_folder, f'error_descriptions_to_feat_{task}.pkl'), 'wb') as fp:
                pickle.dump(self.error_descriptions_to_feat, fp)




        # graph input
        if self.use_gcn:
            with open(os.path.join(root_dir, 'active_object.json'), 'r') as fp:
                all_active_obj = json.load(fp)
            
            active_obj = all_active_obj[task]
            self.bboxes = {}
            self.bbox_classes = {}
            self.edge_maps = {}
            for i in range(len(active_obj)):
                video_id = active_obj[i]['video_id']
                if video_id in self.data_list:
                    object_info = active_obj[i]['active_obj']
                    bbox_class, bbox, edge_map = generate_node_connected(object_info, num_node, height, width)
                    self.bboxes[video_id] = bbox
                    self.bbox_classes[video_id] = bbox_class
                    self.edge_maps[video_id] = edge_map

    def __len__(self):
        if self.online_test:
            return len(self.data_list_online_test)
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        if self.online_test:
            video_id, end_seg_idx = self.data_list_online_test[idx]
        else:
            video_id = self.data_list[idx]
        
        annots = self.annotations[video_id]
        time_stamps, action_labels, action_labels_error, error_descriptions = annots


        error_descriptions_feat = []
        for time_stamp, error_description in zip(time_stamps, error_descriptions):
            start_frame = int(time_stamp[0])
            end_frame = int(time_stamp[1])
            error_descriptions_feat = error_descriptions_feat + [self.error_descriptions_to_feat[error_description]] * (end_frame - start_frame + 1)

        error_descriptions_feat = torch.cat(error_descriptions_feat, dim=0)
        
        feats = np.load(os.path.join(self.feat_path, video_id+'.npy'))

        # ignore some background segments
        if self.is_training:
            delete_idx = []
            for i in range(len(action_labels)):
                if action_labels[i] == self.bg_idx and random.random() > self.background_ratio:
                    delete_idx.append(i)
            if len(delete_idx) != 0:
                time_stamps = np.delete(time_stamps, delete_idx, 0)
                action_labels = np.delete(action_labels, delete_idx, 0)
                action_labels_error = np.delete(action_labels_error, delete_idx, 0)

        data_dict = {
            'feats': torch.from_numpy(feats).permute(1, 0).float(),
            'error_descriptions_feat': error_descriptions_feat,
            'segments': torch.from_numpy(time_stamps).float(),
            'labels': torch.from_numpy(action_labels).long(),
            'labels_error': torch.from_numpy(action_labels_error).long(),
            'video_id': str(video_id),
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
            'action_id_to_str': self.action_id_to_str,
            'error_descriptions_to_feat': self.error_descriptions_to_feat
        }

        
        if self.use_gcn:
            bbox_class = self.bbox_classes[video_id]
            bbox = self.bboxes[video_id]
            edge_map = self.edge_maps[video_id]
            data_dict['bbox_class'] = torch.tensor(bbox_class).long()
            data_dict['bbox'] = torch.tensor(bbox).float()
            data_dict['edge_map'] = torch.tensor(edge_map).float()


        # truncate the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)
        elif self.online_test:
            data_dict = truncate_online_data(data_dict, end_seg_idx)


        # print(data_dict['segments'])
        # print(data_dict['labels'])
        # print(data_dict['labels_error'])
        # assert False

        return data_dict


