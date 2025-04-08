import json
from collections import defaultdict
import os
from typing import List, Dict, Set, Tuple
import heapq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DAGBuilder:
    def __init__(self, json_path: str):
        """初始化DAG构建器
        Args:
            json_path: 序列数据的JSON文件路径
        """
        # 加载数据
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.sequences = data['sequences']
        self.label_to_idx = data['label_to_idx']
        self.idx_to_label = data['idx_to_label']
        self.task = data['task']
        self.segment_type = data['segment_type']
        self.end_node = data['end_node']
        
        # 统计边的权重
        self.edge_weights = self._count_edge_weights()
        
    def _count_edge_weights(self) -> Dict[Tuple[int, int], int]:
        """统计所有边的权重"""
        weights = defaultdict(int)
        for seq in self.sequences:
            for i in range(len(seq)-1):
                weights[(seq[i], seq[i+1])] += 1
        return weights
    
    def _has_cycle(self, graph: Dict[int, List[int]], start: int, 
                   visited: Set[int], path: Set[int]) -> bool:
        """检查是否存在环"""
        visited.add(start)
        path.add(start)
        
        for next_node in graph.get(start, []):
            if next_node not in visited:
                if self._has_cycle(graph, next_node, visited, path):
                    return True
            elif next_node in path:
                return True
                
        path.remove(start)
        return False
    
    def build_optimal_dag(self, weight_threshold: int = 5) -> List[Tuple[int, int, int]]:
        """构建最优的DAG
        Args:
            weight_threshold: 边权重阈值，低于此值的边将被过滤
        Returns:
            List[Tuple[int, int, int]]: 列表of (from_node, to_node, weight)
        """
        # 按权重降序排序边，并过滤低权重的边
        edges = [(w, src, dst) for (src, dst), w in self.edge_weights.items() 
                 if w >= weight_threshold]
        edges.sort(reverse=True)  # 按权重降序
        
        # 构建DAG
        final_edges = []
        graph = defaultdict(list)
        nodes = {0, self.end_node}  # 确保包含起始和结束节点
        
        for weight, src, dst in edges:
            # 临时添加边
            graph[src].append(dst)
            nodes.add(src)
            nodes.add(dst)
            
            # 检查是否形成环
            visited = set()
            path = set()
            
            has_cycle = False
            for node in nodes:
                if node not in visited:
                    if self._has_cycle(graph, node, visited, path):
                        has_cycle = True
                        break
            
            # 如果形成环，移除该边
            if has_cycle:
                graph[src].remove(dst)
            else:
                final_edges.append((src, dst, weight))
        
        return final_edges
    
    def save_dag(self, edges: List[Tuple[int, int, int]], save_path: str):
        """保存DAG到文件
        Args:
            edges: DAG的边列表
            save_path: 保存路径
        """
        # 构建保存的数据结构
        dag_data = {
            'task': self.task,
            'segment_type': self.segment_type,
            'end_node': self.end_node,
            'edges': edges,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }
        
        # 保存到文件
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(dag_data, f, indent=2)
    
    def print_dag_stats(self, edges: List[Tuple[int, int, int]]):
        """打印DAG的统计信息"""
        print(f"\nDAG Statistics for {self.task} ({self.segment_type}):")
        print(f"Total edges: {len(edges)}")
        
        # 统计每个节点的出度和入度
        out_degree = defaultdict(int)
        in_degree = defaultdict(int)
        total_weight = 0
        
        for src, dst, weight in edges:
            out_degree[src] += 1
            in_degree[dst] += 1
            total_weight += weight
        
        print(f"Total weight: {total_weight}")
        print(f"Number of nodes: {len(set(out_degree.keys()) | set(in_degree.keys()))}")
        print(f"Start node (0) out degree: {out_degree[0]}")
        print(f"End node ({self.end_node}) in degree: {in_degree[self.end_node]}")

    def get_transition_matrix(self, edges: List[Tuple[int, int, int]]) -> np.ndarray:
        """生成转移矩阵
        Args:
            edges: DAG的边列表
        Returns:
            np.ndarray: 转移矩阵
        """
        # 获取节点数量（包括起始节点0和结束节点）
        n_nodes = self.end_node + 1
        
        # 初始化转移矩阵
        trans_matrix = np.zeros((n_nodes, n_nodes))
        
        # 填充转移矩阵
        for src, dst, weight in edges:
            trans_matrix[src, dst] = weight
            
        return trans_matrix
    
    def plot_heatmap(self, edges: List[Tuple[int, int, int]], save_path: str):
        """绘制并保存转移矩阵热力图
        Args:
            edges: DAG的边列表
            save_path: 保存路径
        """
        # 获取转移矩阵
        trans_matrix = self.get_transition_matrix(edges)
        
        # 创建标签列表（将数字映射到动作名称）
        labels = []
        for i in range(self.end_node + 1):
            if i == 0:
                labels.append("START")
            elif i == self.end_node:
                labels.append("END")
            else:
                labels.append(self.idx_to_label.get(str(i), str(i)))
        
        # 设置图形大小
        plt.figure(figsize=(15, 12))
        
        # 绘制热力图
        sns.heatmap(trans_matrix, 
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='YlOrRd',  # 使用YlOrRd配色方案
                   annot=True,     # 显示数值
                   fmt='g',        # 整数格式
                   square=True,    # 正方形单元格
                   cbar_kws={'label': 'Transition Count'})
        
        # 设置标题
        plt.title(f'Transition Matrix Heatmap - {self.task} ({self.segment_type})')
        
        # 调整布局
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_task_graph(self, edges: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
        """将带权重的边转换为简单的任务图格式
        Args:
            edges: DAG的边列表 (from_node, to_node, weight)
        Returns:
            List[Tuple[int, int]]: 任务图格式的边列表 (from_node, to_node)
        """
        return [(src, dst) for src, dst, _ in edges]

def main():
    # 处理所有任务
    tasks = ['coffee', 'atv', 'belt', 'circuitbreaker', 'computer', 
             'dslr', 'gladom_assemble', 'gladom_disassemble', 'gopro',
             'knarrevik_assemble', 'knarrevik_disassemble', 'marius_assemble',
             'marius_disassemble', 'navvis', 'nespresso', 'printer_big',
             'printer_small', 'rashult_assemble', 'rashult_disassemble', 'switch']
    
    # 创建保存目录
    os.makedirs('./dag_buffer', exist_ok=True)
    os.makedirs('./dag_buffer/heatmaps', exist_ok=True)
    
    # 存储所有任务的图
    graphs_dict = {}
    
    # 设置权重阈值
    weight_threshold = 5
    
    for task in tasks:
        for segment_type in ['n', 'v']:
            # 加载序列数据
            seq_path = f'./holoassist_sequences_buffer/sequences_{task}_{segment_type}.json'
            if not os.path.exists(seq_path):
                continue
                
            print(f"\nProcessing {task} ({segment_type})...")
            
            # 构建DAG
            builder = DAGBuilder(seq_path)
            edges = builder.build_optimal_dag(weight_threshold=weight_threshold)
            
            # 转换为任务图格式并存储
            task_key = f"{task}_{segment_type}"
            graphs_dict[task_key] = builder.get_task_graph(edges)
            
            # 保存DAG
            dag_path = f'./holoassist_dag_buffer/dag_{task}_{segment_type}.json'
            builder.save_dag(edges, dag_path)
            
            # 生成并保存热力图
            heatmap_path = f'./holoassist_dag_buffer/heatmaps/{task}_{segment_type}_heatmap.png'
            builder.plot_heatmap(edges, heatmap_path)
            
            # 打印统计信息
            builder.print_dag_stats(edges)
    
    # 保存任务图字典
    with open('./holoassist_dag_buffer/task_graphs.json', 'w') as f:
        json.dump(graphs_dict, f)
    
    print("\nTask graphs have been saved to task_graphs.json")
    return graphs_dict

if __name__ == '__main__':
    graphs = main()
