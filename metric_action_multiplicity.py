import json
import numpy as np
import torch
import logging
import os
from sklearn.metrics import accuracy_score
import pickle
import argparse
from multiprocessing import Pool
import multiprocessing as mp

# 已定义的多样性节点字典
action_diverse_nodes_dict = {
    'coffee': [0, 1, 2, 3, 5, 8, 9, 11, 12, 13, 15],
    'pinwheels': [4, 8, 12],
    'oatmeal': [0, 1, 3, 6, 8, 11],
    'quesadilla': [3, 4, 5],
    'tea': [0, 1, 2, 3, 4, 5, 6]
}

class ActionMultiplicityEvaluator:
    def __init__(self, args):
        self.args = args
        self.annotations = {}
        
        # 读取标注数据
        with open(args.annotation_path, 'r') as f:
            all_annot = json.load(f)
        
        # 读取测试集列表
        with open(os.path.join(args.root_dir, args.task, 'test.txt'), 'r') as fp:
            self.data_list = [line.strip('\n') for line in fp]
            
        # 处理标注数据
        annot = all_annot[args.task]
        for i in range(len(annot['segments'])):
            video_id = annot['segments'][i]['video_id']
            if video_id in self.data_list:
                actions = [int(action) for action in annot['segments'][i]['labels']['action']]
                action_types = [int(action_type) for action_type in annot['segments'][i]['labels']['action_type']]
                self.annotations[video_id] = [
                    np.array(annot['segments'][i]['labels']['time_stamp']) * args.fps,
                    np.array(actions),
                    np.array(action_types)
                ]
        
        # 获取当前任务的多样性节点
        self.diverse_nodes = action_diverse_nodes_dict[args.task]
        
        # 修改获取quantiles的逻辑
        temp_path = os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_0.00.pkl")
        with open(temp_path, 'rb') as f:
            results = pickle.load(f)
        video_id = next(iter(results))
        
        # 检查数据结构是否包含quantile
        if 'quantile_labels' in results[video_id]:
            self.has_quantile = True
            self.quantiles = [key for key in results[video_id]['quantile_labels'].keys()]
        else:
            self.has_quantile = False
            self.quantiles = [None]  # 只进行一次评估

    def evaluate_after_diverse(self, threshold=None, quantile=None):
        """评估多样性节点之后的节点的性能"""
        
        # 读取预测结果
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, f'pred_seg_results_{threshold:.2f}.pkl'), 'rb') as f:
            results = pickle.load(f)
            
        all_after_diverse_acc_original = []
        all_after_diverse_acc_corrected = []
        all_after_diverse_error_acc = []
        all_after_diverse_error_only_acc = []
        all_normal_acc_original = []
        all_normal_acc_corrected = []
        all_normal_error_acc = []
        all_normal_error_only_acc = []
        
        for video_id in self.data_list:
            # 获取ground truth和预测结果
            gt_segments, gt_labels, gt_error_types = self.annotations[video_id]
            pred_segments = results[video_id]['segments']
            
            # 根据数据结构选择标签
            if self.has_quantile:
                if quantile is not None:
                    pred_error_labels = results[video_id]['quantile_labels'][quantile]
                else:
                    pred_error_labels = results[video_id]['label']
            else:
                # 对于旧的数据结构，尝试不同的键名
                if 'labels' in results[video_id]:
                    pred_error_labels = results[video_id]['labels']
                elif 'label' in results[video_id]:
                    pred_error_labels = results[video_id]['label']
                else:
                    raise KeyError(f"Cannot find labels in results for video {video_id}")
                
            # 获取original和corrected标签
            # 对于旧数据结构，使用相同的预测标签
            original_labels = pred_error_labels
            corrected_labels = pred_error_labels
            
            if self.has_quantile:
                # 对于新数据结构，尝试获取original和corrected标签
                original_labels = results[video_id].get('original_labels', pred_error_labels)
                corrected_labels = results[video_id].get('corrected_labels', pred_error_labels)
            
            # 转换为帧级别的标签
            length = int(gt_segments[-1, 1])
            gt_frames = self.to_frame_wise(gt_segments, gt_labels, length)
            gt_error_frames = self.to_frame_wise(gt_segments, gt_error_types, length)
            pred_error_frames = self.to_frame_wise(pred_segments, pred_error_labels, length)
            pred_original_frames = self.to_frame_wise(pred_segments, original_labels, length)
            pred_corrected_frames = self.to_frame_wise(pred_segments, corrected_labels, length)
            
            # 分析每个节点
            for i in range(0, len(gt_labels)):
                prev_action = gt_labels[i-1] if i > 0 else None
                curr_start = int(gt_segments[i][0])
                curr_end = int(gt_segments[i][1])
                
                curr_gt = gt_frames[curr_start:curr_end+1]
                curr_pred_error = pred_error_frames[curr_start:curr_end+1]
                curr_gt_error = gt_error_frames[curr_start:curr_end+1]
                curr_pred_original = pred_original_frames[curr_start:curr_end+1]
                curr_pred_corrected = pred_corrected_frames[curr_start:curr_end+1]
                
                # 计算准确率
                action_acc_original = accuracy_score(curr_gt, curr_pred_original)
                action_acc_corrected = accuracy_score(curr_gt, curr_pred_corrected)
                
                # 计算整体error detection准确率（包括正确和错误的帧）
                error_acc = np.mean(
                    ((curr_gt_error > 0) & (curr_pred_error == -1)) |
                    ((curr_gt_error == 0) & (curr_pred_error != -1))
                )
                
                # 修改：只计算gt为error的情况下的准确率
                error_indices = curr_gt_error > 0
                if np.any(error_indices):
                    # 只在gt为error的位置计算预测正确的比例
                    error_only_acc = np.sum((curr_gt_error > 0) & (curr_pred_error == -1)) / np.sum(curr_gt_error > 0)
                else:
                    error_only_acc = 0  # 如果没有error帧，设为0
                
                if prev_action in self.diverse_nodes:
                    all_after_diverse_acc_original.append(action_acc_original)
                    all_after_diverse_acc_corrected.append(action_acc_corrected)
                    all_after_diverse_error_acc.append(error_acc)
                    all_after_diverse_error_only_acc.append(error_only_acc)
                else:
                    all_normal_acc_original.append(action_acc_original)
                    all_normal_acc_corrected.append(action_acc_corrected)
                    all_normal_error_acc.append(error_acc)
                    all_normal_error_only_acc.append(error_only_acc)
        
        # 计算结果
        after_diverse_acc_original = np.mean(all_after_diverse_acc_original) if all_after_diverse_acc_original else 0
        after_diverse_acc_corrected = np.mean(all_after_diverse_acc_corrected) if all_after_diverse_acc_corrected else 0
        after_diverse_error_acc = np.mean(all_after_diverse_error_acc) if all_after_diverse_error_acc else 0
        after_diverse_error_only_acc = np.mean(all_after_diverse_error_only_acc) if all_after_diverse_error_only_acc else 0
        normal_acc_original = np.mean(all_normal_acc_original) if all_normal_acc_original else 0
        normal_acc_corrected = np.mean(all_normal_acc_corrected) if all_normal_acc_corrected else 0
        normal_error_acc = np.mean(all_normal_error_acc) if all_normal_error_acc else 0
        normal_error_only_acc = np.mean(all_normal_error_only_acc) if all_normal_error_only_acc else 0
        
        # 输出结果
        if quantile is not None:
            logging.info(f"Task: {self.args.task}, Threshold: {threshold:.2f}, Quantile: {quantile}")
        else:
            logging.info(f"Task: {self.args.task}, Threshold: {threshold:.2f}")
        logging.info(f"After diverse nodes - Original Action Acc: {after_diverse_acc_original:.3f}, Corrected Action Acc: {after_diverse_acc_corrected:.3f}, Error Detection Acc: {after_diverse_error_acc:.3f}, Error-Only Detection Acc: {after_diverse_error_only_acc:.3f}")
        logging.info(f"After normal nodes - Original Action Acc: {normal_acc_original:.3f}, Corrected Action Acc: {normal_acc_corrected:.3f}, Error Detection Acc: {normal_error_acc:.3f}, Error-Only Detection Acc: {normal_error_only_acc:.3f}")
        
        return {
            'after_diverse_acc_original': after_diverse_acc_original,
            'after_diverse_acc_corrected': after_diverse_acc_corrected,
            'after_diverse_error_acc': after_diverse_error_acc,
            'after_diverse_error_only_acc': after_diverse_error_only_acc,
            'normal_acc_original': normal_acc_original,
            'normal_acc_corrected': normal_acc_corrected,
            'normal_error_acc': normal_error_acc,
            'normal_error_only_acc': normal_error_only_acc
        }

    @staticmethod
    def to_frame_wise(segments, labels, length):
        """将段级别的标签转换为帧级别"""
        frames = np.zeros(length)
        for i, ((start, end), label) in enumerate(zip(segments, labels)):
            frames[int(start):int(end)+1] = label
        return frames

def process_threshold(args_combination, args):
    threshold, quantile = args_combination
    result = {}
    result['threshold'] = threshold
    result['quantile'] = quantile
    
    # 获取评估结果
    metrics = evaluator.evaluate_after_diverse(threshold=threshold, quantile=quantile)
    result.update(metrics)
    return result

def init_evaluator(args):
    global evaluator
    evaluator = ActionMultiplicityEvaluator(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='EgoPER')
    # parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--fps', default=10, type=int)
    parser.add_argument('--root_dir', type=str, default='./EgoPER')
    parser.add_argument('--annotation_path', type=str, default='./EgoPER/annotation.json')
    parser.add_argument('--processes', type=int, default=16)
    args = parser.parse_args()
    
    # 设置日志
    config_name = args.dirname.rsplit('_', 1)[0]
    log_subfolder = f'./logs/{config_name}'
    os.makedirs(log_subfolder, exist_ok=True)
    log_file = f'{log_subfolder}/metric_action_multiplicity.log'
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # 从dirname中提取task
    if 'coffee' in args.dirname:
        args.task = 'coffee'
    elif 'pinwheels' in args.dirname:
        args.task = 'pinwheels'
    elif 'oatmeal' in args.dirname:
        args.task = 'oatmeal'
    elif 'quesadilla' in args.dirname:
        args.task = 'quesadilla'
    elif 'tea' in args.dirname:
        args.task = 'tea'
    else:
        raise ValueError(f"Cannot determine task from dirname: {args.dirname}")
    
    # 初始化评估器
    evaluator = ActionMultiplicityEvaluator(args)
    
    # 设置阈值范围和quantiles
    thresholds = [i / 10 for i in range(-20, 21)]  # -2.0 到 2.0
    args_combinations = [(threshold, quantile) for threshold in thresholds for quantile in evaluator.quantiles]
    results_by_quantile = {quantile: [] for quantile in evaluator.quantiles}
    
    # 多进程处理不同阈值
    if args.processes <= 0:
        # 单进程模式
        results = [process_threshold(args_combination, args) for args_combination in args_combinations]
    else:
        # 多进程模式
        with Pool(processes=args.processes, initializer=init_evaluator, initargs=(args,)) as pool:
            results = pool.starmap(process_threshold, [(args_combination, args) for args_combination in args_combinations])
    
    # 将结果按quantile分组
    for result in results:
        quantile = result['quantile']
        results_by_quantile[quantile].append(result)
    
    # 对每个quantile处理结果
    for quantile, results in results_by_quantile.items():
        # 收集所有指标
        after_diverse_metrics = {
            'acc_original': [],
            'acc_corrected': [],
            'error_acc': [],
            'error_only_acc': []
        }
        normal_metrics = {
            'acc_original': [],
            'acc_corrected': [],
            'error_acc': [],
            'error_only_acc': []
        }
        
        # 收集结果
        for result in results:
            after_diverse_metrics['acc_original'].append(result['after_diverse_acc_original'])
            after_diverse_metrics['acc_corrected'].append(result['after_diverse_acc_corrected'])
            after_diverse_metrics['error_acc'].append(result['after_diverse_error_acc'])
            after_diverse_metrics['error_only_acc'].append(result['after_diverse_error_only_acc'])
            normal_metrics['acc_original'].append(result['normal_acc_original'])
            normal_metrics['acc_corrected'].append(result['normal_acc_corrected'])
            normal_metrics['error_acc'].append(result['normal_error_acc'])
            normal_metrics['error_only_acc'].append(result['normal_error_only_acc'])
        
        # 计算平均值和最大值
        after_diverse_avg = {k: np.mean(v) for k, v in after_diverse_metrics.items()}
        normal_avg = {k: np.mean(v) for k, v in normal_metrics.items()}
        after_diverse_max = {k: np.max(v) for k, v in after_diverse_metrics.items()}
        normal_max = {k: np.max(v) for k, v in normal_metrics.items()}
        
        # 保存结果
        save_path = os.path.join('./ckpt', args.dataset, args.dirname)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'after_diverse_metrics_quantile_{quantile}.npy'), 
                np.array([after_diverse_metrics['acc_original'], 
                         after_diverse_metrics['acc_corrected'],
                         after_diverse_metrics['error_acc'],
                         after_diverse_metrics['error_only_acc']]))
        np.save(os.path.join(save_path, f'normal_metrics_quantile_{quantile}.npy'), 
                np.array([normal_metrics['acc_original'], 
                         normal_metrics['acc_corrected'],
                         normal_metrics['error_acc'],
                         normal_metrics['error_only_acc']]))
        
        # 输出结果
        logging.info(f'\nQuantile: {quantile}')
        logging.info('After diverse nodes:')
        logging.info(f'Average Original Acc: {after_diverse_avg["acc_original"]:.3f}')
        logging.info(f'Average Corrected Acc: {after_diverse_avg["acc_corrected"]:.3f}')
        logging.info(f'Average Error Detection Acc: {after_diverse_avg["error_acc"]:.3f}')
        logging.info(f'Average Error-Only Detection Acc: {after_diverse_avg["error_only_acc"]:.3f}')
        logging.info(f'Max Original Acc: {after_diverse_max["acc_original"]:.3f}')
        logging.info(f'Max Corrected Acc: {after_diverse_max["acc_corrected"]:.3f}')
        logging.info(f'Max Error Detection Acc: {after_diverse_max["error_acc"]:.3f}')
        logging.info(f'Max Error-Only Detection Acc: {after_diverse_max["error_only_acc"]:.3f}')
        
        logging.info('After normal nodes:')
        logging.info(f'Average Original Acc: {normal_avg["acc_original"]:.3f}')
        logging.info(f'Average Corrected Acc: {normal_avg["acc_corrected"]:.3f}')
        logging.info(f'Average Error Detection Acc: {normal_avg["error_acc"]:.3f}')
        logging.info(f'Average Error-Only Detection Acc: {normal_avg["error_only_acc"]:.3f}')
        logging.info(f'Max Original Acc: {normal_max["acc_original"]:.3f}')
        logging.info(f'Max Corrected Acc: {normal_max["acc_corrected"]:.3f}')
        logging.info(f'Max Error Detection Acc: {normal_max["error_acc"]:.3f}')
        logging.info(f'Max Error-Only Detection Acc: {normal_max["error_only_acc"]:.3f}')

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    main()
