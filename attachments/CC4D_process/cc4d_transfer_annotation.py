import json
import numpy as np

def collect_step_info(json_files):
    """从多个json文件中收集step信息，只收集action id的映射"""
    activity_step_maps = {}
    all_steps = set()  # 用于收集所有activity的step_id
    
    # 遍历所有json文件
    for json_file_path in json_files:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 收集每个activity的所有step_id
        for record in data:
            activity_id = record['activity_id']
            if activity_id not in activity_step_maps:
                activity_step_maps[activity_id] = set()
                
            for step in record['step_annotations']:
                old_step_id = step['step_id']
                if old_step_id != 0:  # 排除背景类
                    activity_step_maps[activity_id].add(old_step_id)
                    all_steps.add(old_step_id)  # 添加到总类的step集合
    
    # 为activity 0（总类）创建映射
    activity_step_maps[0] = set(all_steps)
    
    # 创建统一的映射关系
    for activity_id in activity_step_maps:
        # 获取排序后的原始step_id列表
        original_steps = sorted(list(activity_step_maps[activity_id]))
        # 创建映射字典
        activity_step_maps[activity_id] = {
            orig_id: new_id 
            for new_id, orig_id in enumerate(original_steps, 1)  # 从1开始编号
        }
    
    return activity_step_maps

def process_json_data(json_file_path, activity_step_maps, fps=10):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 初始化处理后的数据字典
    processed_data = {}
    processed_data[0] = {}  # 初始化总类
    
    # 遍历每个记录
    for record in data:
        activity_id = record['activity_id']
        recording_id = record['recording_id']
        
        # 如果activity_id不在processed_data中，添加新的字典
        if activity_id not in processed_data:
            processed_data[activity_id] = {}
            
        # 初始化当前记录的数据结构
        segments = []
        labels = []
        labels_error = []
        descriptions = []
        
        # 获取所有有效的步骤（start_time不为-1）并按开始时间排序
        valid_steps = [step for step in record['step_annotations'] if step['start_time'] != -1.0]
        valid_steps.sort(key=lambda x: x['start_time'])
        
        # 如果没有有效步骤，跳过这条记录
        if not valid_steps:
            print(f"No valid steps found for activity_id: {activity_id}, recording_id: {recording_id}")
            continue
            
        # 处理第一个动作之前的背景（如果有）
        first_start = valid_steps[0]['start_time']
        if first_start > 0:
            segments.append([0, int(first_start * fps) - 1])
            labels.append(0)
            labels_error.append(0)
            descriptions.append("background")
        
        # 处理每个步骤和步骤之间的间隔
        for i, step in enumerate(valid_steps):
            # 添加当前步骤
            start_frame = int(step['start_time'] * fps)
            end_frame = int(step['end_time'] * fps)
            segments.append([start_frame, end_frame])
            
            # 将原始step_id映射到新的id
            old_step_id = step['step_id']
            new_step_id = activity_step_maps[activity_id].get(old_step_id, 0)
            labels.append(new_step_id)
            
            # 同时获取总类的映射id
            new_step_id_all = activity_step_maps[0].get(old_step_id, 0)
            
            has_error = -1 if 'errors' in step else 0
            labels_error.append(has_error)
            
            # 直接使用当前step的description
            if 'modified_description' in step:
                descriptions.append(step['modified_description'])
            else:
                descriptions.append(step['description'])
            
            # 如果不是最后一个步骤，添加与下一个步骤之间的背景
            if i < len(valid_steps) - 1:
                next_start = valid_steps[i + 1]['start_time']
                if next_start > step['end_time']:
                    segments.append([end_frame + 1, int(next_start * fps) - 1])
                    labels.append(0)
                    labels_error.append(0)
                    descriptions.append("background")
        
        # 将处理后的数据添加到字典中
        if segments:
            # 添加到原始activity
            processed_data[activity_id][recording_id] = {
                'segments': segments,
                'labels': labels,
                'labels_error': labels_error,
                'descriptions': descriptions
            }
            
            # 添加到总类（activity 0）
            # 创建总类的labels（使用总类的映射）
            all_labels = []
            for label in labels:
                if label == 0:  # 如果是背景类
                    all_labels.append(0)
                else:
                    # 找到原始step_id
                    for orig_id, new_id in activity_step_maps[activity_id].items():
                        if new_id == label:
                            # 使用总类的映射
                            all_labels.append(activity_step_maps[0][orig_id])
                            break
            
            processed_data[0][f"0_{recording_id}"] = {
                'segments': segments,
                'labels': all_labels,
                'labels_error': labels_error,
                'descriptions': descriptions
            }
    
    return processed_data

if __name__ == "__main__":
    # 定义输入输出文件路径
    json_files = ["non_error_samples.json", "error_samples.json"]
    
    # 首先从所有数据集收集step信息
    activity_step_maps = collect_step_info(json_files)
    
    # 保存统一的映射表
    mapping_data = {
        'step_id_mapping': activity_step_maps
    }
    with open('step_id_mapping.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print("Step ID mapping has been saved to step_id_mapping.json")
    
    # 处理每个数据集
    for json_file in json_files:
        output_file = json_file.replace('.json', '_processed.json')
        processed_data = process_json_data(json_file, activity_step_maps)
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=4)
        print(f"Processed data has been saved to {output_file}")
    
