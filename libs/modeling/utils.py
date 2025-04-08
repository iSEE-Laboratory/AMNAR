import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from libs.datasets.data_utils import to_frame_wise, to_segments


def calculate_distance(tensor1, tensor2, distance_type='L2', pairwise=False, use_norm=False):
    """
    计算两个张量之间的距离，支持两两计算返回[N, M]矩阵的距离。

    参数：
    tensor1 (torch.Tensor): 大小为 [C] 或 [N, C] 的张量。
    tensor2 (torch.Tensor): 大小为 [C] 或 [M, C] 的另一个张量。
    distance_type (str): 距离类型，选项有 'L1', 'L2', 'cosine'。默认为 'L2'。
    pairwise (bool): 如果为 True，计算两两之间的距离，返回 [N, M] 矩阵；否则计算对应元素之间的距离。
    use_norm (bool): 如果为 True，对输入张量进行归一化处理。默认为 True。

    返回：
    torch.Tensor: 表示计算出的距离的张量，大小为 []（标量）、[N] 或 [N, M]。
    """
    # print(f"use_norm: {use_norm}")
    # assert use_norm == True
    # 确保 tensor1 和 tensor2 至少是 2D
    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(0)

    if pairwise:
        # 计算两两之间的距离
        if use_norm:
            tensor1 = torch.nn.functional.normalize(tensor1, p=2, dim=-1)
            tensor2 = torch.nn.functional.normalize(tensor2, p=2, dim=-1)

        if distance_type == 'L1':
            distance = torch.cdist(tensor1, tensor2, p=1)
        elif distance_type == 'L2':
            distance = torch.cdist(tensor1, tensor2, p=2)
        elif distance_type == 'cosine':
            cosine_sim = torch.mm(tensor1, tensor2.t())
            distance = 1 - cosine_sim
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")
    else:
        # 计算对应元素之间的距离
        if tensor1.size() != tensor2.size():
            raise ValueError("When pairwise is False, tensor1 and tensor2 must have the same size.")
        
        if use_norm:
            tensor1 = torch.nn.functional.normalize(tensor1, p=2, dim=-1)
            tensor2 = torch.nn.functional.normalize(tensor2, p=2, dim=-1)

        if distance_type == 'L1':
            distance = torch.sum(torch.abs(tensor1 - tensor2), dim=-1)
        elif distance_type == 'L2':
            distance = torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=-1))
        elif distance_type == 'cosine':
            cosine_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)
            distance = 1 - cosine_sim
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

    return distance.squeeze()


def transform_frame_feat_to_action_feat(frame_feats, mode='mean'):
    '''
    frame_feats: [B, C, T] or [C, T]
    mode: 'max', 'mean'
    '''
    # 检查输入张量的维度
    if frame_feats.dim() == 3:
        # 输入为 [B, C, T] 的情况
        if mode == 'max':
            action_feat, _ = torch.max(frame_feats, dim=-1)
        elif mode == 'mean':
            action_feat = torch.mean(frame_feats, dim=-1)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    elif frame_feats.dim() == 2:
        # 输入为 [C, T] 的情况
        if mode == 'max':
            action_feat, _ = torch.max(frame_feats.unsqueeze(0), dim=-1)
            action_feat = action_feat.squeeze(0)
        elif mode == 'mean':
            action_feat = torch.mean(frame_feats.unsqueeze(0), dim=-1).squeeze(0)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    else:
        raise ValueError(f"Unexpected input dimensions: {frame_feats.shape}")
    
    return action_feat

    
class LinearTransformModel(nn.Module):
    def __init__(self, input_dim, output_dim, return_transformed=False, use_norm=True):
        super(LinearTransformModel, self).__init__()
        self.return_transformed = return_transformed
        self.use_norm = use_norm
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, output_dim)
        )
        # self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        if self.return_transformed:
            x = self.linear(x)

        # if self.use_norm:
        #     x = self.layer_norm(x)

        return x


class ActionClusters:
    def __init__(self, num_classes, use_norm=False, distance_type='L2'):
        self.num_classes = num_classes
        self.clusters_feats = {i:[] for i in range(num_classes)}
        self.clusters_centers = {i:None for i in range(num_classes)}
        self.clusters_radius = {i:None for i in range(num_classes)}
        self.centers_weights = {}
        self.centers_distances = {}
        
        # 新增一个字典用于存储测试集的特征
        self.test_clusters_feats = {i:[] for i in range(num_classes)}
        self.test_error_clusters_feats = {i:[] for i in range(num_classes)}
        self.use_norm = use_norm
        self.distance_type = distance_type


    def add_feats(self, feats, labels):
        for feat, label in zip(feats, labels):
            if isinstance(label, torch.Tensor):
                label = int(label)
            self.clusters_feats[label].append(feat)

    def clear_feats(self):
        self.clusters_feats = {i:[] for i in range(self.num_classes)}
        self.test_clusters_feats = {i:[] for i in range(self.num_classes)}
        self.test_error_clusters_feats = {i:[] for i in range(self.num_classes)}

    # 新增函数,用于添加测试集的特征  
    def add_test_feats(self, feats, labels, is_error):
        # 检查是否已经声明了self.test_clusters_feats和self.test_error_clusters_feats
        if not hasattr(self, 'test_clusters_feats'):
            self.test_clusters_feats = {i:[] for i in range(self.num_classes)}
        if not hasattr(self, 'test_error_clusters_feats'):
            self.test_error_clusters_feats = {i:[] for i in range(self.num_classes)}

        # 选择要添加到哪个字典
        target_dict = self.test_error_clusters_feats if is_error else self.test_clusters_feats
        for feat, label in zip(feats, labels):
            if isinstance(label, torch.Tensor):
                label = int(label)
            target_dict[label].append(feat)

    def update_clusters(self, cluster_center_update_momentum=0.8):
        for label in range(self.num_classes):
            if len(self.clusters_feats[label]) == 0:
                if self.clusters_centers[label] is not None:
                    continue
                else:
                    if self.clusters_centers[0] is not None:
                        self.clusters_centers[label] = self.clusters_centers[0].clone()
                        logging.warning(f"Cluster {label} has no features, use the center of cluster 0 instead.")
                    else:
                        valid_centers = [center for center in self.clusters_centers.values() if center is not None]
                        if valid_centers:
                            self.clusters_centers[label] = torch.mean(torch.stack(valid_centers, dim=0), dim=0)
                            logging.warning(f"Cluster {label} has no features, use the mean of all valid centers instead.")
                        else:
                            self.clusters_centers[label] = None
                            logging.warning(f"Cluster {label} has no features, and no valid centers found.")
                    continue
            feats = torch.stack(self.clusters_feats[label], dim=0)
            
            # 使用平均值作为聚类中心
            current_center = torch.mean(feats, dim=0)
            if self.clusters_centers[label] is not None:
                self.clusters_centers[label] = cluster_center_update_momentum * self.clusters_centers[label] + (1 - cluster_center_update_momentum) * current_center
            else:
                self.clusters_centers[label] = current_center
            
            # 计算所有样本到聚类中心的距离
            center = self.clusters_centers[label]
            stack_cluster_center = center.unsqueeze(0).expand(feats.shape[0], -1)
            distances = calculate_distance(feats, stack_cluster_center, use_norm=self.use_norm, distance_type=self.distance_type)
            if distances.ndim == 0:
                distances = distances.unsqueeze(0)
            distances_sorted, _ = torch.sort(distances)
            
            # 计算覆盖 90%, 95% 和 100% 样本的半径
            num_samples = distances.numel()
            if num_samples == 0:
                continue
            elif num_samples == 1:
                idx_90 = 0
                idx_95 = 0
                idx_100 = 0
            else:
                idx_90 = int(num_samples * 0.9) - 1
                idx_95 = int(num_samples * 0.95) - 1
                idx_100 = num_samples - 1  # 100% 覆盖即最大半径
            self.clusters_radius[label] = [
                distances_sorted[idx_90].item(),
                distances_sorted[idx_95].item(),
                distances_sorted[idx_100].item()
            ]
            
            # 计算落在不同半径内的样本比例
            for i, radius in enumerate(self.clusters_radius[label]):
                num_within_radius = torch.sum(distances <= radius).item()
                num_total = num_samples
                ratio_within_radius = num_within_radius / num_total
                logging.info(f"Class {label} ratio within {i}th radius ({radius:.4f}): {num_within_radius}/{num_total} ({ratio_within_radius:.4f})")

        # 计算聚类中心之间的距离
        centers = [self.clusters_centers[label] for label in range(self.num_classes) if self.clusters_centers[label] is not None]
        num_centers = len(centers)
        for i in range(num_centers):
            for j in range(i + 1, num_centers):
                distance = calculate_distance(centers[i], centers[j], use_norm=self.use_norm)
                self.centers_distances[(i, j)] = distance
        
        # 打印聚类中心之间的距离
        logging.info("Distances between cluster centers:")
        for (i, j), distance in self.centers_distances.items():
            logging.info(f"Distance between center {i} and center {j}: {distance:.4f}")


    def calculate_distances_and_weights(self):
        # 确保所有中心已计算s
        centers = [self.clusters_centers[label] for label in self.clusters_centers if self.clusters_centers[label] is not None]
        n_centers = len(centers)
        if n_centers < 2:
            return  # 至少需要两个聚类中心来计算距离
        
        # 计算中心之间的距离
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                distance = calculate_distance(centers[i], centers[j], use_norm=self.use_norm, distance_type=self.distance_type)
                self.centers_distances[(i, j)] = distance
                weight = 1 / distance if distance != 0 else 0
                self.centers_weights[(i, j)] = weight


    def visualize_clusters(self):

        all_feats = torch.cat([torch.stack(feats) for feats in self.clusters_feats.values() if feats], dim=0)
        all_labels = torch.cat([torch.full((len(feats),), label) for label, feats in enumerate(self.clusters_feats.values()) if feats])

        all_feats_numpy = all_feats.cpu().detach().numpy()
        all_labels_numpy = all_labels.cpu().detach().numpy()

        # 去除标签为0的数据
        mask = all_labels_numpy != 0
        filtered_feats = all_feats_numpy[mask]
        filtered_labels = all_labels_numpy[mask]

        # 如果过滤后没有数据，直接返回
        if filtered_feats.shape[0] == 0:
            logging.info("No data available after filtering label 0.")
            return

        # 计算原始特征的轮廓分数
        original_silhouette_score = silhouette_score(filtered_feats, filtered_labels)
        logging.info("Filtered Silhouette Score:", original_silhouette_score)

        # 应用t-SNE进行特征降维
        tsne = TSNE(n_components=3, random_state=42)
        tsne_feats = tsne.fit_transform(filtered_feats)

        # 计算过滤后数据的聚类中心
        centers_tsne = np.array([np.mean(tsne_feats[filtered_labels == i], axis=0) for i in np.unique(filtered_labels)])

        fig = go.Figure()

        # 添加散点图数据
        fig.add_trace(go.Scatter3d(
            x=tsne_feats[:, 0],
            y=tsne_feats[:, 1],
            z=tsne_feats[:, 2],
            mode='markers',
            marker=dict(size=5, color=filtered_labels, colorscale='Viridis', opacity=0.8),
            text=filtered_labels
        ))

        # 添加聚类中心
        fig.add_trace(go.Scatter3d(
            x=centers_tsne[:, 0],
            y=centers_tsne[:, 1],
            z=centers_tsne[:, 2],
            mode='markers',
            marker=dict(size=10, color='red'),
            text=['Center' for _ in centers_tsne],
            name='Centers'
        ))

        # 设置图形布局
        fig.update_layout(
            title="3D Visualization of Clustered Features with t-SNE (Excluding Label 0)",
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        fig.write_html('tsne_cluster_visualization_excluding_label_0.html')

    def visualize_distances_heatmap(self):
        n_centers = len(self.clusters_centers)
        distance_matrix = np.zeros((n_centers, n_centers))
        for (i, j), distance in self.centers_distances.items():
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Distance Heatmap between Cluster Centers")
        plt.savefig("cluster_centers_distances_heatmap.png")
        plt.close()


    def evaluate_test_set(self):
        def process_test_set(test_clusters_feats, set_name):
            test_feats = torch.cat([torch.stack(feats) for feats in test_clusters_feats.values() if feats], dim=0)
            test_labels = torch.cat([torch.full((len(feats),), label) for label, feats in enumerate(test_clusters_feats.values()) if feats])

            # 计算每个测试样本到所有聚类中心的距离
            centers = torch.stack([self.clusters_centers[i] for i in range(self.num_classes) if self.clusters_centers[i] is not None])
            # 计算每个测试样本到所有聚类中心的距离
            distances = torch.zeros(test_feats.shape[0], centers.shape[0])
            for i in range(centers.shape[0]):
                center = centers[i]
                stack_center = center.unsqueeze(0).expand(test_feats.shape[0], -1)
                center_distances = calculate_distance(test_feats, stack_center, use_norm=self.use_norm, distance_type=self.distance_type)
                distances[:, i] = center_distances

            # 准备数据用于绘图
            class_ratios = [[] for _ in range(self.num_classes)]  # 为每个类别创建一个空列表
            class_radii = [[] for _ in range(self.num_classes)]   # 为每个类别创建一个空列表

            # 计算每个类别的测试样本到聚类中心的距离是否小于半径
            for i in range(self.num_classes):
                if len(test_clusters_feats[i]) > 0:
                    class_distances = distances[test_labels == i, i]
                    radii = torch.tensor(self.clusters_radius[i])
                    
                    ratios = []
                    for j, radius in enumerate(radii):
                        within_radius = (class_distances <= radius)
                        num_within_radius = within_radius.sum().item()
                        num_total = len(class_distances)
                        ratio = within_radius.float().mean()
                        ratios.append(ratio.item())
                        logging.info(f"{set_name} - Class {i} ratio within {j}th radius ({radius:.4f}): {num_within_radius}/{num_total} ({ratio:.4f})")
                        
                        # 打印超出半径的样本信息
                        outside_radius_indices = torch.where(~within_radius)[0]
                        for idx in outside_radius_indices:
                            sample_distance = class_distances[idx].item()
                            logging.info(f"Sample {idx.item()} - Class: {i}, Distance: {sample_distance:.4f}, Radius: {radius:.4f}")
                    
                    class_ratios[i] = ratios
                    class_radii[i] = radii.tolist()
                else:
                    # 如果类别没有样本,则将准确率设为0
                    class_ratios[i] = [0] * len(self.clusters_radius[i])
                    class_radii[i] = self.clusters_radius[i]

            # 绘制柱状图
            num_classes = len(class_ratios)
            num_radii = len(class_ratios[0])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(num_classes)
            width = 0.25
            
            for i in range(num_radii):
                ratios = [class_ratio[i] for class_ratio in class_ratios]
                ax.bar(x + i*width, ratios, width, label=f'{i}th radius')
            
            ax.set_ylabel('Accuracy (Ratio within radius)')
            ax.set_xlabel('Class')
            ax.set_title(f'Accuracy within Different Radii for Each Class - {set_name}')
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'class_accuracy_visualization_{set_name}.png')
            plt.close()

            # 打印每个半径的具体值
            for i, radii in enumerate(class_radii):
                logging.info(f"{set_name} - Class {i} radii: {', '.join([f'{r:.4f}' for r in radii])}")

        # 处理正确的测试集
        logging.info("Evaluating correct test set:")
        process_test_set(self.test_clusters_feats, "Correct")

        # 处理错误的测试集
        logging.info("\nEvaluating error test set:")
        process_test_set(self.test_error_clusters_feats, "Error")


def most_common_element(lst):
    if not lst:
        return None  # Handle empty list
    count_dict = {}
    for item in lst:
        count_dict[item] = count_dict.get(item, 0) + 1
    return max(count_dict, key=count_dict.get)  # Return the key with the maximum count

def calculate_intersection_ratio(pred_start, pred_end, gt_start, gt_end):
    # Calculate the intersection of two segments
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection_length = max(0, intersection_end - intersection_start + 1)
    pred_length = pred_end - pred_start + 1
    return intersection_length / pred_length


def filter_segments_and_labels(single_video_gt, single_result, intersection_threshold=0.5):
    gt_segments = single_video_gt['segments']
    gt_labels = single_video_gt['labels']
    
    length = int(gt_segments[-1][1]) + 1

    pred_segments = single_result['segments']
    pred_labels = single_result['labels']
    pred_scores = single_result['scores']
    _, pred_segments_frame = to_segments(to_frame_wise(pred_segments, pred_labels, pred_scores, length, fps=single_result['fps']))

    gt_labels_frame = to_frame_wise(gt_segments, gt_labels, None, length)

    filtered_segments = []
    filtered_labels = []

    for pred_start, pred_end in pred_segments_frame:
        max_intersection_ratio = 0
        majority_label = 0
        # 找到与当前预测segment交集最大的真实segment
        for gt_start, gt_end in gt_segments:
            intersection_ratio = calculate_intersection_ratio(pred_start, pred_end, gt_start, gt_end)
            if intersection_ratio >= max_intersection_ratio:
                max_intersection_ratio = intersection_ratio
                # 取交集内的真实标签的众数作为该预测segment的标签
                majority_label = most_common_element(gt_labels_frame[pred_start:pred_end + 1].tolist())

        if max_intersection_ratio >= intersection_threshold:
            # 如果最大交集比例大于阈值,则认为该预测segment有效
            filtered_segments.append((pred_start, pred_end))
            filtered_labels.append(majority_label)

    return filtered_segments, filtered_labels