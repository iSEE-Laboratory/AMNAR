import os
import torch
import pickle
import numpy as np
import json
import argparse
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import auc
from libs.datasets.data_utils import to_frame_wise, to_segments
from eval_utils import Video, Checkpoint, eval_omission_error
from copy import deepcopy
from multiprocessing import Pool
import multiprocessing as mp
import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='To copy construct from a tensor*')
# loose version
def error_acc(pred, gt, gt_error):
    num_correct = 0
    num_total = 0
    num_error = 0
    num_nonerror = 0 
    pre_gt = None
    pre_gt_error = None
    is_pred_error = False
    for i in range(len(gt)):
        if pre_gt is None:
            pre_gt = gt[i]
            is_pre_gt_error = True if gt_error[i] == -1 else False

        if pred[i] == -1:
            is_pred_error = True
            num_error += 1
        else:
            num_nonerror += 1

        if pre_gt != gt[i]:
            if is_pred_error and is_pre_gt_error and num_error > num_nonerror:
                num_correct += 1
            # elif (not is_pred_error and not is_pre_gt_error) or num_error < num_nonerror:
            elif (not is_pred_error and not is_pre_gt_error) or (num_error < num_nonerror and not is_pre_gt_error):
                num_correct += 1
            is_pred_error = False
            pre_gt = gt[i]
            is_pre_gt_error = True if gt_error[i] == -1 else False
            num_error = 0
            num_nonerror = 0
            num_total += 1

    if is_pred_error and is_pre_gt_error and num_error > num_nonerror:
        num_correct += 1
    elif (not is_pred_error and not is_pre_gt_error) or num_error < num_nonerror:
        num_correct += 1
    num_total += 1

    return num_correct, num_total

def acc_tpr_fpr(all_preds, all_gts):
    # fpr = fp / (fp + tn)
    all_gt_normal = all_preds[all_gts == 1] # get predicted non-error items
    fp_tn = len(all_gt_normal) # number of total non-error items in the ground truth
    fp = len(all_gt_normal[all_gt_normal == -1]) # get FP
    
    # tpr = tp / (tp + fn)
    all_gt_error = all_preds[all_gts == -1] # get predicted error items
    tp_fn = len(all_gt_error) # number of total error items in the ground truth
    tp = len(all_gt_error[all_gt_error == -1]) # get TP

    # acc
    acc = torch.eq(torch.LongTensor(all_gts), torch.LongTensor(all_preds)).sum() / len(all_gts)

    if tp_fn == 0:
        if tp == 0:
            tpr = 1
        else:
            tpr = 0
    else:
        tpr = tp / tp_fn

    if fp_tn == 0:
        if fp == 0:
            fpr = 1
        else:
            fpr = 0
    else:
        fpr = fp / fp_tn
    
    return acc, tpr, fpr

def acc_precision_recall_f1(all_preds, all_gts, set_labels, each_class = True):
    if each_class:
        method = None
        acc = None
        for j in set_labels:
            each_acc = torch.eq(torch.LongTensor(all_gts[all_gts == j]), torch.LongTensor(all_preds[all_gts == j])).sum() / len(all_gts[all_gts == j])
            if acc is None:
                acc = each_acc.unsqueeze(0)
            else:
                acc = torch.cat((acc, each_acc.unsqueeze(0)), dim=0)
    else:
        method = 'macro'
        acc = torch.eq(torch.LongTensor(all_gts), torch.LongTensor(all_preds)).sum() / len(all_gts)
    p = precision_score(all_gts, all_preds, labels=set_labels, average=method,zero_division=0)
    r = recall_score(all_gts, all_preds, labels=set_labels, average=method,zero_division=0)

    f1 = 2 * p * r / (p + r)
    
    return acc, p, r, f1

def generate_partitions(inputs):
    cur_class = None
    start = 0
    step_partitions = []
    for i in range(len(inputs)):
        if inputs[i] != cur_class and cur_class is not None:
            step_partitions.append((cur_class, i - start + 1))
            start = i + 1
        cur_class = inputs[i]
    step_partitions.append((inputs[len(inputs) - 1], len(inputs) - start + 1))
    return step_partitions

def pred_vis(all_gts, all_preds, mapping, vname, category_colors=None):
    clean_version = False  # Set to False to add labels
    gts = all_gts
    preds = all_preds

    if category_colors is None:
        mycmap = plt.get_cmap('rainbow', len(mapping))
        category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]

    # Generate ground truth segments
    gt_partitions = generate_partitions(gts)
          
    plt.figure(figsize=(25, 4))
    
    # Ground truth visualization
    plt.subplot(211)
    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        rects = plt.barh('gt_segmentation', w, left=data_cum, height=0.5, color=category_colors[l.item()])
        text_color = 'black'
        data_cum += w

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=1, bbox_to_anchor=(1, -1), loc='lower left', fontsize='small')

    # Predictions visualization
    plt.subplot(212)
    pred_partitions = generate_partitions(preds)
    data_cum = 0
    for i, (l, w) in enumerate(pred_partitions):
        rects = plt.barh('pred_segmentation', w, left=data_cum, height=0.5, color=category_colors[l.item()])
        text_color = 'black'
        data_cum += w
        
    plt.savefig(f'./{vname}.jpg')
    plt.close()

class ActionSegmentationErrorDetectionEvaluator:
    def __init__(self, args):
        self.args = args
        self.annotations = {}
        self.online_test = args.online_test
        # self.step_annotations = {}
        task = args.task
        EgoPER_root_dir = './EgoPER'
        HoloAssist_root_dir = './HoloAssist'
        CaptainCook4D_root_dir = './CaptainCook4D'
        CaptainCook4D_feat_dir = 'feats/I3D'
        
        if args.dataset == 'EgoPER':
            with open(os.path.join(EgoPER_root_dir, args.task, 'test.txt'), 'r') as fp:
                lines = fp.readlines()
                self.data_list = [line.strip('\n') for line in lines]
            with open(os.path.join(EgoPER_root_dir, 'annotation.json'), 'r') as fp:
                all_annot = json.load(fp)
        elif args.dataset == 'HoloAssist':
            testset_path = os.path.join(HoloAssist_root_dir, 'data-task-splits/val', f'{args.task}.txt')
            with open(testset_path, 'r') as fp:
                lines = fp.readlines()
                self.data_list = [line.strip('\n') for line in lines]
            with open(os.path.join(f'libs/datasets/holoassist_processed_annotations_{args.segment_type}.json'), 'r') as fp:
                all_annot = json.load(fp)
        elif args.dataset == 'CaptainCook4D':
            with open(os.path.join(CaptainCook4D_root_dir, 'CC4D_process', 'error_samples_processed.json'), 'r') as fp:
                all_annot = json.load(fp)

        self.data_list_online_test = []
 
        annot = all_annot[task]
        

        if args.dataset == 'EgoPER':
            action2idx = annot['action2idx']
        elif args.dataset == 'HoloAssist':
            assert args.segment_type is not None
            label_dict_path = os.path.join('dataset_buffer', f'holoassist_label_dict_{args.task}_{args.segment_type}.pkl')
            with open(label_dict_path, 'rb') as fp:
                label_dict = pickle.load(fp)
            action2idx = label_dict['label_to_idx']
        elif args.dataset == 'CaptainCook4D':
            with open(os.path.join(CaptainCook4D_root_dir, 'CC4D_process', 'step_id_mapping.json'), 'r') as fp:
                step_id_mapping = json.load(fp)['step_id_mapping']
            action2idx = step_id_mapping[task]
            action2idx['0'] = 0



        if args.dataset == 'EgoPER':
            for i in range(len(annot['segments'])):
                video_id = annot['segments'][i]['video_id']
                if video_id in self.data_list:
                    actions = [int(action) for action in annot['segments'][i]['labels']['action']]
                    num_seg = len(actions)
                    for j in range(num_seg):
                        self.data_list_online_test.append((video_id, j))
                    action_types = [int(action_type) for action_type in annot['segments'][i]['labels']['action_type']]
                    self.annotations[video_id] = [np.array(annot['segments'][i]['labels']['time_stamp']) * args.fps, 
                                                np.array(actions),
                                                np.array(action_types),
                                                annot['segments'][i]['labels']['error_description']]
        elif args.dataset == 'HoloAssist':
            for sample_name, sample_annot in annot.items():
                if sample_name in self.data_list:
                    # Convert string labels to numeric indices
                    labels_idx = np.array([action2idx.get(label, 0) for label in sample_annot['labels']])
                    # Adapt metric code, convert -1 to 1, keep 0 unchanged
                    labels_error = np.array(sample_annot['labels_error'])
                    labels_error[labels_error == -1] = 1
                    self.annotations[sample_name] = [np.array(sample_annot['segments']),
                                                   labels_idx,  # Use converted numeric indices
                                                   np.array(sample_annot['labels_error']),
                                                   np.array(sample_annot['vn_pairs'])]
        elif args.dataset == 'CaptainCook4D':
            for sample_name, sample_annot in annot.items():
                self.annotations[sample_name] = [np.array(sample_annot['segments']),
                                                np.array(sample_annot['labels']),
                                                np.array(sample_annot['labels_error']),
                                                np.array(sample_annot['descriptions'])]
        
        
        if args.dataset == 'CaptainCook4D':
            self.data_list = []
            for video_id, _ in self.annotations.items():
                if args.task == '0':
                    video_id_real = video_id.split('_', 1)[1]
                else:
                    video_id_real = video_id
                feat_path = os.path.join(CaptainCook4D_root_dir, CaptainCook4D_feat_dir, f'{video_id_real}_360p.npy')
                if os.path.exists(feat_path):
                    self.data_list.append(video_id)
                else:
                    logging.info(f"Filtered out videos not in annotations: {video_id}")
        else:
            # Filter out videos not in annotations
            filtered_videos = [video_id for video_id in self.data_list if video_id not in self.annotations]
            if filtered_videos:
                logging.info(f"Filtered out videos not in annotations: {filtered_videos}")
            self.data_list = [video_id for video_id in self.data_list if video_id in self.annotations]


        if args.error_detection:
            # find all quantiles
            temp_path = os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_0.00.pkl")
            with open(temp_path, 'rb') as f:
                results = pickle.load(f)
            video_id = next(iter(results))  # Get any key from the dictionary
            self.quantiles = [key for key in results[video_id]['quantile_labels'].keys()]
        



        self.idx2action = {}
        for key, value in action2idx.items():
            self.idx2action[int(value)] = key
        self.set_labels = [i for i in range(len(action2idx))]
        # import ipdb; ipdb.set_trace()

    # EDA
    def macro_segment_error_detection(self, output_list=None, threshold=None, quantile=None):
        def main_process(video_id, gt_segments, gt_labels, results, total_correct, total, eng_seg_idx=None):
            length = int(gt_segments[-1, 1])

            gt = to_frame_wise(gt_segments, gt_labels, None, length)
            gt_error = to_frame_wise(gt_segments, gt_label_types, None, length)

            # convert all error types to -1
            gt_error[gt_error > 0] = -1
            gt_error[gt_error == 0] = 1
            
            segments = torch.tensor(results[video_id]['segments']).clone().detach()
            if quantile is not None:
                labels = torch.tensor(results[video_id]['quantile_labels'][quantile]).clone().detach()
            else:
                labels = torch.tensor(results[video_id]['label']).clone().detach()
            scores = torch.tensor(results[video_id]['score']).clone().detach()

            pred = to_frame_wise(segments, labels, None, length)

            # convert all normal classes to 1
            pred[pred >= 0] = 1
            
            if self.online_test:
                end_seg = gt_segments[eng_seg_idx]
                gt = gt[int(end_seg[0]):int(end_seg[1]) + 1]
                pred = pred[int(end_seg[0]):int(end_seg[1]) + 1]
                gt_error = gt_error[int(end_seg[0]):int(end_seg[1]) + 1]

            num_correct, num_total = error_acc(pred, gt, gt_error)

            total_correct += num_correct
            total += num_total

            return total_correct, total

        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        total_correct = 0
        total = 0
        if self.online_test:
            for video_id, eng_seg_idx in self.data_list_online_test:
                video_id_with_end_seg_idx = f'{video_id}_{eng_seg_idx}'
                gt_segments, gt_labels, gt_label_types, gt_des = deepcopy(self.annotations[video_id])
                
                total_correct, total = main_process(video_id_with_end_seg_idx, gt_segments, gt_labels, results, total_correct, total, eng_seg_idx)
        else:
            for video_id in self.data_list:
                
                gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
                
                total_correct, total = main_process(video_id, gt_segments, gt_labels, results, total_correct, total)
            

        if output_list is not None:
            output_list.append(total_correct / total)
            return output_list
        else:
            logging.info("|Error detection accuracy|%.3f|"%(total_correct / total))

    # Micro AUC
    def micro_framewise_error_detection(self, output_list=None, threshold=None, quantile=None, is_visualize=False):
        def handle_visualization(gt_error, pred, video_id):
            category_colors = {
                1: '#ABD5A5',  # RGB(171, 213, 165) - Normal action
                -1: '#F18C84'  # RGB(241, 140, 132) - Error action
            }
            
            dir_path = os.path.join('./visualization/', self.args.dataset, self.args.dirname)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            quantile_dir = os.path.join(dir_path, f'quantile{quantile:.2f}')
            if not os.path.exists(quantile_dir):
                os.makedirs(quantile_dir, exist_ok=True)

            if -2.0 <= threshold <= 2.0:
                threshold_dir = os.path.join(quantile_dir, f'threshold{threshold:.1f}')
                if not os.path.exists(threshold_dir):
                    os.mkdir(threshold_dir)
                pred_vis(gt_error, pred, self.idx2action, os.path.join(threshold_dir, f'ed_{video_id}'), category_colors=category_colors)

        def main_process(video_id, gt_segments, results, preds, gts, eng_seg_idx=None):
            length = int(gt_segments[-1][1])

            gt_error = to_frame_wise(gt_segments, gt_label_types, None, length)

            # convert all error types to -1
            gt_error[gt_error > 0] = -1
            gt_error[gt_error == 0] = 1

            segments = torch.tensor(results[video_id]['segments']).clone().detach()
            if quantile is not None:
                labels = torch.tensor(results[video_id]['quantile_labels'][quantile]).clone().detach()
            else:
                labels = torch.tensor(results[video_id]['label']).clone().detach()

            pred = to_frame_wise(segments, labels, None, length)

            # convert all normal classes to 1
            pred[pred >= 0] = 1

            if self.online_test:
                end_seg = gt_segments[eng_seg_idx]
                gt_error = gt_error[int(end_seg[0]):int(end_seg[1]) + 1]
                pred = pred[int(end_seg[0]):int(end_seg[1]) + 1]

            if preds is None:
                preds = pred
                gts = gt_error
            else:
                preds = torch.cat((preds, pred), dim=0)
                gts = torch.cat((gts, gt_error), dim=0)

            return gt_error, pred, video_id, preds, gts


        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        preds = None
        gts = None

        if self.online_test:
            for video_id, eng_seg_idx in self.data_list_online_test:
                video_id_with_end_seg_idx = f'{video_id}_{eng_seg_idx}'
                gt_segments, gt_labels, gt_label_types, gt_des = deepcopy(self.annotations[video_id])
                results = deepcopy(results)
                
                gt_error, pred, video_id, preds, gts = main_process(video_id_with_end_seg_idx, gt_segments, results, preds, gts, eng_seg_idx)

                if is_visualize:
                    handle_visualization(gt_error, pred, video_id)
        else:
            for video_id in self.data_list:
                
                gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
                
                gt_error, pred, video_id, preds, gts = main_process(video_id, gt_segments, results, preds, gts)

                if is_visualize:
                    handle_visualization(gt_error, pred, video_id)
        
        final_acc, final_tpr, final_fpr = acc_tpr_fpr(preds, gts)

        if output_list is not None:
            output_list['acc'].append(final_acc)
            output_list['tpr'].append(final_tpr)
            output_list['fpr'].append(final_fpr)
            return output_list
        else:
            logging.info("|Error detection (thres=%.3f, micro)|acc=%.3f|fpr=%.3f|tpr=%.3f|"%(threshold, final_acc, final_fpr, final_tpr))
        
    # Macro AUC
    def macro_framewise_error_detection(self, output_list=None, threshold=None, quantile=None, is_visualize=False):
        
        def main_process(video_id, gt_segments, results, eng_seg_idx=None):
            length = int(gt_segments[-1, 1])

            gt_error = to_frame_wise(gt_segments, gt_label_types, None, length)
            
            # convert all error types to -1
            gt_error[gt_error > 0] = -1
            gt_error[gt_error == 0] = 1

            segments = torch.tensor(results[video_id]['segments']).clone().detach()
            if quantile is not None:
                labels = torch.tensor(results[video_id]['quantile_labels'][quantile]).clone().detach()
            else:
                labels = torch.tensor(results[video_id]['label']).clone().detach()
            scores = torch.tensor(results[video_id]['score']).clone().detach()

            pred = to_frame_wise(segments, labels, None, length)
            
            # convert all normal classes to 1
            pred[pred >= 0] = 1

            if self.online_test:
                end_seg = gt_segments[eng_seg_idx]
                gt_error = gt_error[int(end_seg[0]):int(end_seg[1]) + 1]
                pred = pred[int(end_seg[0]):int(end_seg[1]) + 1]


            acc, tpr, fpr = acc_tpr_fpr(pred, gt_error)

            acc_list.append(acc)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

            return acc_list, tpr_list, fpr_list



        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        preds = None
        gts = None
        acc_list = []
        tpr_list = []
        fpr_list = []

        if self.online_test:
            for video_id, eng_seg_idx in self.data_list_online_test:
                video_id_with_end_seg_idx = f'{video_id}_{eng_seg_idx}'
                gt_segments, gt_labels, gt_label_types, gt_des = deepcopy(self.annotations[video_id])
                results = deepcopy(results)

                acc_list, tpr_list, fpr_list = main_process(video_id_with_end_seg_idx, gt_segments, results, eng_seg_idx)
        else:
            for video_id in self.data_list:
                
                gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
                
                acc_list, tpr_list, fpr_list = main_process(video_id, gt_segments, results)
        
        final_acc = np.array(acc_list).mean()
        final_tpr = np.array(tpr_list).mean()
        final_fpr = np.array(fpr_list).mean()

        if output_list is not None:
            output_list['acc'].append(final_acc)
            output_list['tpr'].append(final_tpr)
            output_list['fpr'].append(final_fpr)
            return output_list
        else:
            logging.info("|Error detection (thres=%.3f, macro)|acc=%.3f|fpr=%.3f|tpr=%.3f|"%(threshold, final_acc, final_fpr, final_tpr))

    # Accuracy, Precision, Recall F1 of Action Segmentation
    def micro_framewise_action_segmentation(self, eval_each_class=True, is_visualize=False):

        def main_process(video_id, gt_segments, gt_labels, results, preds, gts, is_visualize, eng_seg_idx=None):
            length = int(gt_segments[-1, 1])
            gt = to_frame_wise(gt_segments, gt_labels, None, length)

            segments = torch.tensor(results[video_id]['segments']).clone().detach()
            labels = torch.tensor(results[video_id]['label']).clone().detach() 
            pred = to_frame_wise(segments, labels, None, length)

            if self.online_test:
                end_seg = gt_segments[eng_seg_idx]
                gt = gt[int(end_seg[0]):int(end_seg[1]) + 1]
                pred = pred[int(end_seg[0]):int(end_seg[1]) + 1]

            if preds is None:
                preds = pred
                gts = gt
            else:
                preds = torch.cat((preds, pred), dim=0)
                gts = torch.cat((gts, gt), dim=0)

            if is_visualize:
                save_dir_path = os.path.join('./visualization/', self.args.dataset, self.args.dirname)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path, exist_ok=True)
                cp_gt = np.copy(gt)
                cp_pred = np.copy(pred)
                pred_vis(cp_gt, cp_pred, self.idx2action, os.path.join(save_dir_path, 'asch_'+ video_id), category_colors=None)

            return preds, gts
        

        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results_list = [pickle.load(f)]

        if self.args.error_detection:
            temp_path = os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_0.00.pkl")
            with open(temp_path, "rb") as f:
                results_ed = pickle.load(f)
            if all("corrected_labels" in data for data in results_ed.values()):
                for video_id, data in results_ed.items():
                    data['label'] = data['corrected_labels']
                results_list.append(results_ed)


        for idx, results in enumerate(results_list):
            if idx == 1:
                logging.info(f'ED Results')
            else:
                logging.info(f'AS Results')
            preds = None
            gts = None

            if self.online_test:
                for video_id, eng_seg_idx in self.data_list_online_test:
                    video_id_with_end_seg_idx = f'{video_id}_{eng_seg_idx}'
                    gt_segments, gt_labels, gt_label_types, gt_des = deepcopy(self.annotations[video_id])

                    preds, gts = main_process(video_id_with_end_seg_idx, gt_segments, gt_labels, results, preds, gts, is_visualize, eng_seg_idx)

            else:
                for video_id in self.data_list:
                    
                    gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
                    
                    preds, gts = main_process(video_id, gt_segments, gt_labels, results, preds, gts, is_visualize)

            
            acc, p, r, f1 = acc_precision_recall_f1(preds, gts, self.set_labels, eval_each_class)
            if eval_each_class:
                for j in range(len(self.set_labels)):
                    logging.info("|Action segmentation (cls head, class %d)|precision %.3f|recall %.3f|f1 %.3f|acc %.3f|"%(self.set_labels[j], p[j], r[j], f1[j], acc[j]))
            else:
                logging.info("|Action segmentation (cls head, macro)|precision %.3f|recall %.3f|f1 %.3f|acc %.3f|"%(p, r, f1, acc))
        
    # IoU, Edit distance, F1@0.5, Accuracy of Action Segmentation
    def standard_action_segmentation(self):
        
        def main_process(video_id, gt_segments, gt_labels, results, input_video_list, eng_seg_idx=None):
            length = int(gt_segments[-1, 1])

            gt = to_frame_wise(gt_segments, gt_labels, None, length)

            segments = torch.tensor(results[video_id]['segments']).clone().detach()
            labels = torch.tensor(results[video_id]['label']).clone().detach()

            pred = to_frame_wise(segments, labels, None, length)

            if self.online_test:
                end_seg = gt_segments[eng_seg_idx]
                gt = gt[int(end_seg[0]):int(end_seg[1]) + 1]
                pred = pred[int(end_seg[0]):int(end_seg[1]) + 1]
            
            input_video = Video(video_id, pred.tolist(), gt.tolist())
            input_video_list.append(input_video)

            return input_video_list

        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)
        
        input_video_list = []

        if self.online_test:
            for video_id, eng_seg_idx in self.data_list_online_test:
                video_id_with_end_seg_idx = f'{video_id}_{eng_seg_idx}'
                gt_segments, gt_labels, gt_label_types, gt_des = deepcopy(self.annotations[video_id])

                input_video_list = main_process(video_id_with_end_seg_idx, gt_segments, gt_labels, results, input_video_list, eng_seg_idx)

        else:
            for video_id in self.data_list:
                
                gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
                
                input_video_list = main_process(video_id, gt_segments, gt_labels, results, input_video_list)


        ckpt = Checkpoint(bg_class=[-1])
        ckpt.add_videos(input_video_list)
        out = ckpt.compute_metrics()
        
        logging.info("|Action segmentation|IoU:%.1f|edit:%.1f|F1@0.5:%.1f|Acc:%.1f|"%(out['IoU']*100, out['edit']*100, out['F1@0.50']*100, out['acc']*100))

    # haven't updated
    def omission_detection(self):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)

        all_pred_action_labels = []
        all_gt_action_labels = []

        action_order_path = f'./{self.args.dataset}/action_order.json'
        with open(action_order_path, 'r') as f:
            task_action_order = json.load(f)[self.args.task]

        for video_id in self.data_list:
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            labels = torch.tensor(results[video_id]['label']).clone().detach()
            labels_list = labels.numpy()


            gt_action_order = task_action_order[video_id]


            all_pred_action_labels.append(labels_list)
            all_gt_action_labels.append(gt_action_order)

        eval_omission_error(self.args.task, all_pred_action_labels, all_gt_action_labels)

def init_evaluator(args):
    global evaluator
    evaluator = ActionSegmentationErrorDetectionEvaluator(args)


def process_threshold(args_combination, args):
    threshold, quantile = args_combination
    result = {}
    result['threshold'] = threshold
    result['quantile'] = quantile
    error_micro_list = {'acc': [], 'tpr': [], 'fpr': []}
    error_macro_list = {'acc': [], 'tpr': [], 'fpr': []}
    eda_list = []

    result['error_micro'] = evaluator.micro_framewise_error_detection(output_list=error_micro_list, threshold=threshold, quantile=quantile, is_visualize=args.visualize)
    result['error_macro'] = evaluator.macro_framewise_error_detection(output_list=error_macro_list, threshold=threshold, quantile=quantile)
    result['eda'] = evaluator.macro_segment_error_detection(output_list=eda_list, threshold=threshold, quantile=quantile)
    return result




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str)
    parser.add_argument('--dataset', type=str, default='EgoPER')
    parser.add_argument('--task', type=str, default='pinwheels')
    parser.add_argument('--segment_type', type=str, default=None)
    parser.add_argument('--fps', default=10, type=int)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('-as', '--action-segmentation', action='store_true', help='Evaluate action segmentation using cls head')
    parser.add_argument('-ed', '--error-detection', action='store_true')
    parser.add_argument('-od', '--omission-detection', action='store_true', help='always with flag --error')
    parser.add_argument('--threshold', default=-100.0, type=float, help='If set to 0.0, plot the curve and fine the best threshold')
    parser.add_argument('-vis', '--visualize', action='store_true')
    parser.add_argument('-ot', '--online-test', action='store_true')
    parser.add_argument('-p', '--processes', type=int, default=16, help='Number of processes for multiprocessing. If <= 0, use single process mode')
    
    args = parser.parse_args()

    mp.set_start_method('forkserver')  # or 'forkserver'

    ckpt_subfolder = os.path.join('ckpt', args.dataset, args.dirname)
    if not os.path.exists(ckpt_subfolder):
        os.makedirs(ckpt_subfolder, exist_ok=True)

    config_name = args.dirname.rsplit('_', 1)[0]
    log_subfolder = f'./logs/{config_name}'
    os.makedirs(log_subfolder, exist_ok=True)
    log_file = f'{log_subfolder}/metric_vis_multiprocess.log'  # Specify log file path
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # Add console output to log
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    evaluator = ActionSegmentationErrorDetectionEvaluator(args)

    if args.action_segmentation:
        evaluator.micro_framewise_action_segmentation(eval_each_class=True)
        evaluator.micro_framewise_action_segmentation(eval_each_class=False, is_visualize=args.visualize)
        evaluator.standard_action_segmentation()

    if args.error_detection:
        thresholds = [i / 10 for i in range(-20, 21)]
        args_combinations = [(threshold, quantile) for threshold in thresholds for quantile in evaluator.quantiles]
        results_by_quantile = {quantile: [] for quantile in evaluator.quantiles}

        if args.processes <= 0:
            # Single process mode
            results = [process_threshold(args_combination, args) for args_combination in args_combinations]
        else:
            # Multi-process mode
            with Pool(processes=args.processes, initializer=init_evaluator, initargs=(args,)) as pool:
                results = pool.starmap(process_threshold, [(args_combination, args) for args_combination in args_combinations])
        
        # Group results by quantile
        for result in results:
            quantile = result['quantile']
            results_by_quantile[quantile].append(result)

        # Process results for each quantile separately
        for quantile, results in results_by_quantile.items():
            error_micro_list = {'acc': [], 'tpr': [], 'fpr': []}
            error_macro_list = {'acc': [], 'tpr': [], 'fpr': []}
            eda_list = []

            for result in results:
                error_micro_list['acc'].extend(result['error_micro']['acc'])
                error_micro_list['tpr'].extend(result['error_micro']['tpr'])
                error_micro_list['fpr'].extend(result['error_micro']['fpr'])
                error_macro_list['acc'].extend(result['error_macro']['acc'])
                error_macro_list['tpr'].extend(result['error_macro']['tpr'])
                error_macro_list['fpr'].extend(result['error_macro']['fpr'])
                eda_list.extend(result['eda'])

            micro_fprs = np.array(error_micro_list['fpr'])
            micro_tprs = np.array(error_micro_list['tpr'])
            macro_fprs = np.array(error_macro_list['fpr'])
            macro_tprs = np.array(error_macro_list['tpr'])
            micro_fprs_tprs = [micro_fprs, micro_tprs]
            macro_fprs_tprs = [macro_fprs, macro_tprs]

            np.save(os.path.join('./ckpt', args.dataset, args.dirname, f'micro_fpr_tpr_quantile_{quantile}.npy'), np.array(micro_fprs_tprs))
            np.save(os.path.join('./ckpt', args.dataset, args.dirname, f'macro_fpr_tpr_quantile_{quantile}.npy'), np.array(macro_fprs_tprs))
            np.save(os.path.join('./ckpt', args.dataset, args.dirname, f'eda_quantile_{quantile}.npy'), np.array(eda_list))

            micro_fprs = np.sort(micro_fprs)
            micro_tprs = np.sort(micro_tprs)
            macro_fprs = np.sort(macro_fprs)
            macro_tprs = np.sort(macro_tprs)

            micro_fprs = np.concatenate((micro_fprs, np.array([1.0])), axis=0)
            micro_tprs = np.concatenate((micro_tprs, np.array([1.0])), axis=0)
            macro_fprs = np.concatenate((macro_fprs, np.array([1.0])), axis=0)
            macro_tprs = np.concatenate((macro_tprs, np.array([1.0])), axis=0)

            micro_auc_value = auc(micro_fprs, micro_tprs)
            macro_auc_value = auc(macro_fprs, macro_tprs)

            logging.info(f'Quantile: {quantile}')
            logging.info('|%s|EDA: %.1f|Micro AUC: %.1f|Macro AUC: %.1f|'%(args.dirname, np.array(eda_list).mean() * 100, micro_auc_value * 100, macro_auc_value * 100))

    if args.omission_detection:
        evaluator.omission_detection()

