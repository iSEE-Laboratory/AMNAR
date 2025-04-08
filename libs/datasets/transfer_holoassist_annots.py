import os
import json
from collections import defaultdict
import numpy as np

def process_annotations():
    # 基础路径设置
    root_dir = '/home/weijin/source/MistakeDetection/FAFP/HoloAssist'
    
    # 读取原始标注文件
    with open(os.path.join(root_dir, 'data-annotation-trainval-v1.json'), 'r') as fp:
        all_annot = json.load(fp)
    
    # 获取所有任务列表
    tasks = ['coffee', 'atv', 'belt', 'circuitbreaker', 'computer', 
             'dslr', 'gladom_assemble', 'gladom_disassemble', 'gopro',
             'knarrevik_assemble', 'knarrevik_disassemble', 'marius_assemble',
             'marius_disassemble', 'navvis', 'nespresso', 'printer_big',
             'printer_small', 'rashult_assemble', 'rashult_disassemble', 'switch']
    
    # 创建任务到视频的映射
    task_videos = {}
    for task in tasks:
        videos = []
        # 读取训练集和验证集的视频列表
        for split in ['train', 'val']:
            split_file = os.path.join(root_dir, 'data-task-splits', split, f'{task}.txt')
            if os.path.exists(split_file):
                with open(split_file, 'r') as fp:
                    videos.extend([line.strip() for line in fp.readlines()])
        task_videos[task] = set(videos)
    
    # 创建新的标注结构
    new_annotations_n = defaultdict(dict)  # 按名词分类
    new_annotations_v = defaultdict(dict)  # 按动词分类
    
    # 处理每个样本
    for sample_data in all_annot:
        video_name = sample_data['video_name']
        # 检查特征文件是否存在
        feat_path = os.path.join('/home/weijin/source/MistakeDetection/FAFP/HoloAssist/feats/I3D', 
                                f'{video_name}.npy')
        if not os.path.exists(feat_path):
            print(f"警告：找不到特征文件 {feat_path}，跳过该样本")
            continue
        # 找出这个视频属于哪个任务
        for task, videos in task_videos.items():
            if video_name in videos:
                # 初始化该样本的标注
                sample_annot = {
                    'segments': [],
                    'labels': [],
                    'labels_error': [],
                    'vn_pairs': []
                }
                
                # 处理每个事件
                for event in sample_data['events']:
                    if event['label'] == 'Fine grained action':
                        # 获取动作信息
                        v = event['attributes']['Verb']
                        n = event['attributes']['Noun']
                        start = int(event['start'] * 10)  # 假设default_fps=10
                        end = int(event['end'] * 10)
                        
                        # 检查正确性
                        correctness = event['attributes']['Action Correctness']
                        is_wrong = 'wrong' in correctness.lower()
                        label_error = -1 if is_wrong else 0
                        
                        # 添加到标注中
                        sample_annot['segments'].append([start, end])
                        sample_annot['labels'].append(n)  # 这里使用名词作为标签
                        sample_annot['labels_error'].append(label_error)
                        sample_annot['vn_pairs'].append(f"{v} {n}")
                
                # 合并相邻且标签相同的片段
                if sample_annot['segments']:  # 确保有片段需要处理
                    merged_segments = []
                    merged_labels = []
                    merged_labels_error = []
                    merged_vn_pairs = []
                    
                    current_segment = sample_annot['segments'][0]
                    current_label = sample_annot['labels'][0]
                    current_error = sample_annot['labels_error'][0]
                    current_vn_pair = sample_annot['vn_pairs'][0]
                    
                    for i in range(1, len(sample_annot['segments'])):
                        if (sample_annot['segments'][i][0] - current_segment[1] <= 1 and 
                            sample_annot['labels'][i] == current_label):
                            # 更新当前片段的结束时间
                            current_segment[1] = sample_annot['segments'][i][1]
                            # 如果任一片段有错误，则合并后的片段标记为错误
                            current_error = -1 if current_error == -1 or sample_annot['labels_error'][i] == -1 else 0
                        else:
                            # 保存当前片段并开始新片段
                            merged_segments.append(current_segment)
                            merged_labels.append(current_label)
                            merged_labels_error.append(current_error)
                            merged_vn_pairs.append(current_vn_pair)
                            
                            current_segment = sample_annot['segments'][i]
                            current_label = sample_annot['labels'][i]
                            current_error = sample_annot['labels_error'][i]
                            current_vn_pair = sample_annot['vn_pairs'][i]
                    
                    # 添加最后一个片段
                    merged_segments.append(current_segment)
                    merged_labels.append(current_label)
                    merged_labels_error.append(current_error)
                    merged_vn_pairs.append(current_vn_pair)
                    
                    # 更新原始标注
                    sample_annot['segments'] = merged_segments
                    sample_annot['labels'] = merged_labels
                    sample_annot['labels_error'] = merged_labels_error
                    sample_annot['vn_pairs'] = merged_vn_pairs
                
                # 合并相邻且标签相同的片段后,添加背景片段
                if sample_annot['segments']:  # 确保有片段需要处理
                    # 获取视频总长度
                    feat_path = os.path.join('/home/weijin/source/MistakeDetection/FAFP/HoloAssist/feats/I3D', 
                                           f'{video_name}.npy')
                    frame_length = len(np.load(feat_path))
                    
                    # 添加背景片段
                    bg_start = 0
                    bg_segments = []
                    bg_labels = []
                    bg_labels_error = []
                    bg_vn_pairs = []
                    
                    for segment in sample_annot['segments']:
                        if bg_start < segment[0]:
                            bg_segments.append([bg_start, segment[0]-1])
                            bg_labels.append('bg')  # 添加这一行
                            bg_labels_error.append(0)
                            bg_vn_pairs.append('bg bg')
                        bg_start = segment[1] + 1
                    
                    # 添加最后一个背景片段(如果需要)
                    if bg_start < frame_length:
                        bg_segments.append([bg_start, frame_length-1])
                        bg_labels.append('bg')
                        bg_labels_error.append(0)
                        bg_vn_pairs.append('bg bg')
                    
                    # 将背景片段添加到现有标注中
                    sample_annot['segments'].extend(bg_segments)
                    sample_annot['labels'].extend(bg_labels)
                    sample_annot['labels_error'].extend(bg_labels_error)
                    sample_annot['vn_pairs'].extend(bg_vn_pairs)
                    
                    # 按segment起始时间排序
                    sorted_indexes = sorted(range(len(sample_annot['segments'])), 
                                         key=lambda x: sample_annot['segments'][x][0])
                    
                    # 根据排序索引重新排列所有列表
                    sample_annot['segments'] = [sample_annot['segments'][i] for i in sorted_indexes]
                    sample_annot['labels'] = [sample_annot['labels'][i] for i in sorted_indexes]
                    sample_annot['labels_error'] = [sample_annot['labels_error'][i] for i in sorted_indexes]
                    sample_annot['vn_pairs'] = [sample_annot['vn_pairs'][i] for i in sorted_indexes]

                # 将处理后的标注添加到对应任务中
                new_annotations_n[task][video_name] = sample_annot.copy()
                new_annotations_v[task][video_name] = sample_annot.copy()
                
                # 修改标签为名词或动词
                new_annotations_n[task][video_name]['labels'] = [pair.split(' ')[1] for pair in sample_annot['vn_pairs']]
                new_annotations_v[task][video_name]['labels'] = [pair.split(' ')[0] for pair in sample_annot['vn_pairs']]
                
                break
    
    # 保存新的标注文件(按名词分类)  
    output_path_n = os.path.join('./', 'holoassist_processed_annotations_n.json')
    with open(output_path_n, 'w') as fp:
        json.dump(new_annotations_n, fp, indent=2)

    print(f"已保存处理后的标注(按名词分类)到: {output_path_n}")

    # 保存新的标注文件(按动词分类)
    output_path_v = os.path.join('./', 'holoassist_processed_annotations_v.json')  
    with open(output_path_v, 'w') as fp:
        json.dump(new_annotations_v, fp, indent=2)

    print(f"已保存处理后的标注(按动词分类)到: {output_path_v}")
    
    # 打印一些统计信息
    for task in tasks:
        print(f"{task}: {len(new_annotations_n[task])} 个样本")

if __name__ == '__main__':
    process_annotations()
