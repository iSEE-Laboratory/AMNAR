import json
from collections import defaultdict
import numpy as np

def load_processed_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_action_sequences_by_activity(processed_data):
    """按activity_id提取动作序列"""
    sequences_by_activity = defaultdict(list)
    for activity_id, activity_data in processed_data.items():
        for recording in activity_data.values():
            # 获取非背景类的动作序列
            sequence = [label for label, error in zip(recording['labels'], recording['labels_error']) 
                      if label != 0]  # 排除背景类
            if sequence:
                sequences_by_activity[activity_id].append([0] + sequence)  # 添加起始背景类
    return sequences_by_activity

def build_transition_matrix(sequences):
    """构建转移权重矩阵"""
    transitions = defaultdict(int)
    
    # 统计所有相邻节点对的转移次数
    for seq in sequences:
        for i in range(len(seq)-1):
            for j in range(i+1, len(seq)):
                transitions[(seq[i], seq[j])] += 1
    
    return transitions

def find_maximum_dag(transitions):
    """使用贪心算法找出权重最大的有向无环图"""
    # 将转移对按权重排序
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    
    # 用于检测环的辅助函数
    def has_cycle(graph, start, visited, path):
        visited[start] = True
        path[start] = True
        
        for next_node in graph[start]:
            if not visited[next_node]:
                if has_cycle(graph, next_node, visited, path):
                    return True
            elif path[next_node]:
                return True
                
        path[start] = False
        return False
    
    # 初始化图
    graph = defaultdict(list)
    edges = []
    
    # 逐个添加边，确保不形成环
    for (src, dst), weight in sorted_transitions:
        # 临时添加边
        graph[src].append(dst)
        
        # 检查是否形成环
        visited = defaultdict(bool)
        path = defaultdict(bool)
        has_cycles = has_cycle(graph, src, visited, path)
        
        if has_cycles:
            # 如果形成环，移除这条边
            graph[src].remove(dst)
        else:
            # 如果不形成环，保留这条边
            edges.append([src, dst])
    
    return edges

def main():
    # 加载处理后的数据
    processed_data = load_processed_data('non_error_samples_processed.json')
    
    # 按activity_id提取动作序列
    sequences_by_activity = extract_action_sequences_by_activity(processed_data)
    
    # 为每个activity构建任务图
    task_graphs = {}
    for activity_id, sequences in sequences_by_activity.items():
        # 构建转移权重矩阵
        transitions = build_transition_matrix(sequences)

        print(f"\nActivity {activity_id} Transition Matrix:")
        for (src, dst), weight in transitions.items():
            print(f"({src} -> {dst}): {weight}")
        
        # 找出最大权重DAG
        dag = find_maximum_dag(transitions)
        task_graphs[activity_id] = dag
        print(f"Activity {activity_id} graph edges: {len(dag)}")
    
    # 保存结果
    with open('task_graph.json', 'w') as f:
        json.dump(task_graphs, f, indent=2)
    
    print("\nTask graphs have been generated and saved to task_graph.json")

if __name__ == "__main__":
    main()
