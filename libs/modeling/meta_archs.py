import torch
from torch import nn
from torch.nn import functional as F

from copy import deepcopy
import numpy as np
import math
import random
import logging
import json
from collections import defaultdict

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from ..utils import batched_nms
from libs.datasets.data_utils import to_frame_wise, to_segments
from .rebuild_models import (
    MultimodalLSTMModel,
    MultimodalMLPModel,
    MultimodalAttentionModel,
    VisualAttentionModel,
    VisualLSTMModel,
    VisualMLPModel,
    SimpleMLPModel,
    MultimodalCrossAttentionModel,
    MultimodalTwoStreamsCrossAttentionModel,
    MultimodalSelfCrossAttentionModel
)
from .utils import ActionClusters, LinearTransformModel, calculate_distance, transform_frame_feat_to_action_feat, most_common_element, calculate_intersection_ratio, filter_segments_and_labels


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks, output_intermediate=False):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        intermediate_feats = tuple()  # Store intermediate layer features

        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            cur_intermediate_feats = []  # Store current level intermediate features

            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
                
                if output_intermediate:
                    cur_intermediate_feats.append(cur_out)  # Collect intermediate layer features
            
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )
            
            if output_intermediate:
                intermediate_feats += (cur_intermediate_feats, )  # Add current level intermediate features to tuple

        # fpn_masks remains the same
        if output_intermediate:
            return out_logits, intermediate_feats
        else:
            return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )
        


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )
        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer_CSPL_GCN")
class PtTransformer_CSPL_GCN(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        use_gcn,               # use AOD
        num_node,              # num of node in each frame
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
        rebuild_model_cfg,     # other cfg for rebuilding model
        error_detection_inference_cfg,
        online_mode,   # online mode. Causal.
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        self.rebuild_model_cfg = rebuild_model_cfg
        self.error_detection_inference_cfg = error_detection_inference_cfg

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.cls_loss_weight = train_cfg['cls_loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']
        self.rebuild_loss_weight = train_cfg['rebuild_loss_weight']
        self.lambda_value = 1.0
        self.model_mode = train_cfg['model_mode']
        assert self.model_mode in ['segment', 'rebuild', 'transform']
        self.use_rebuild_expected_feat = (self.model_mode == 'rebuild')
        self.use_transform_feat = (self.model_mode == 'transform')
        self.rebuild_loss_choice = train_cfg['rebuild_loss_choice']
        self.rebuild_model_choice = train_cfg['rebuild_model_choice']
        self.transform_model_choice = train_cfg['transform_model_choice']
        assert self.transform_model_choice in ['linear']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        self.use_gcn = use_gcn
        assert backbone_type in ['convGCNTransformer', 'convTransformer']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                backbone_type,
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'online_mode': online_mode,
                    'use_gcn': use_gcn,
                }
            )
        else:
            self.backbone = make_backbone(
                backbone_type,
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'num_node' : num_node,
                    'online_mode': online_mode,
                    'use_gcn': use_gcn,
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.fpn_type = fpn_type
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        self.fpn_dim = fpn_dim
        self.head_dim = head_dim
        self.head_kernel_size = head_kernel_size
        self.head_with_ln = head_with_ln
        self.head_num_layers = head_num_layers
        self.train_cfg = train_cfg
        self.cosine_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        
        if self.use_transform_feat or self.use_rebuild_expected_feat:
            if self.use_transform_feat:
                return_transformed = True
            else:
                return_transformed = self.rebuild_model_cfg['return_transformed']
                
            if self.transform_model_choice == 'linear':
                self.transform_model = LinearTransformModel(
                    input_dim=fpn_dim,
                    output_dim=fpn_dim,
                    return_transformed=return_transformed
                )

            if self.rebuild_model_cfg['freeze_backbone'] and self.use_rebuild_expected_feat:
                logging.info("Setting backbone, neck, cls_head, and reg_head to eval mode.")
                self.backbone.eval()
                self.neck.eval()
                self.cls_head.eval()
                self.reg_head.eval()

                logging.info("Freezing parameters of backbone, neck, cls_head, and reg_head.")
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False
                for name, param in self.neck.named_parameters():
                    param.requires_grad = False
                for name, param in self.cls_head.named_parameters():
                    param.requires_grad = False
                for name, param in self.reg_head.named_parameters():
                    param.requires_grad = False

            
        
        self.predict_class = self.rebuild_model_cfg['predict_class']
        if self.use_rebuild_expected_feat:
            self.future_step = 1


            if self.rebuild_model_cfg['diversity_mode'] == 'cluster_center':
                text_dim = fpn_dim
            elif self.rebuild_model_cfg['diversity_mode'] == 'error_desc':
                text_dim = 1024
            else:
                raise NotImplementedError('Rebuild model choice not implemented')
            
            common_params = {
                'visual_dim': fpn_dim,
                'text_dim': text_dim,
                'hidden_dim': fpn_dim,
                'output_dim': fpn_dim,
                'num_classes': self.num_classes if self.predict_class else -1,
                'num_layers': rebuild_model_cfg['num_layers'],
                'future_step': self.future_step,
                'win_len': rebuild_model_cfg['win_len'],
                'num_heads': rebuild_model_cfg['num_heads'],
                'concat_query_to_attnout': rebuild_model_cfg['concat_query_to_attnout'],
                'use_layernorm': rebuild_model_cfg['use_layernorm'],
                'drop_rate': rebuild_model_cfg['drop_rate'],
                'prompt_len': rebuild_model_cfg['prompt_len'],
                'noise_std': rebuild_model_cfg['noise_std'],
                'conv_non_linear': rebuild_model_cfg['conv_non_linear']
            }

            if self.rebuild_model_choice == 'lstm':
                self.expected_feat_predictor = MultimodalLSTMModel(**common_params)
            elif self.rebuild_model_choice == 'mlp':
                self.expected_feat_predictor = MultimodalMLPModel(**common_params, history_length=16)
            elif self.rebuild_model_choice == 'attn':
                self.expected_feat_predictor = MultimodalAttentionModel(**common_params)
            elif self.rebuild_model_choice == 'visual_only_attn':
                self.expected_feat_predictor = VisualAttentionModel(**common_params)
            elif self.rebuild_model_choice == 'visual_only_lstm':
                self.expected_feat_predictor = VisualLSTMModel(**common_params)
            elif self.rebuild_model_choice == 'visual_only_mlp':
                self.expected_feat_predictor = VisualMLPModel(**common_params, history_length=16)
            elif self.rebuild_model_choice == 'simple_mlp':
                self.expected_feat_predictor = SimpleMLPModel(**common_params, context_length=16)
            elif self.rebuild_model_choice == 'cross_attn':
                self.expected_feat_predictor = MultimodalCrossAttentionModel(**common_params)
            elif self.rebuild_model_choice == 'self_cross_attn':
                self.expected_feat_predictor = MultimodalSelfCrossAttentionModel(**common_params)
            elif self.rebuild_model_choice == '2s_ca':
                self.expected_feat_predictor = MultimodalTwoStreamsCrossAttentionModel(**common_params)
            else:
                raise NotImplementedError('Rebuild model choice not implemented')
            
            logging.info("Setting transform_model to eval mode.")
            self.transform_model.eval()
            
            if self.rebuild_model_cfg['freeze_backbone'] and self.use_transform_feat:
                logging.info("Freezing parameters of transform_model.")
                for name, param in self.transform_model.named_parameters():
                    param.requires_grad = False
        
        if self.use_transform_feat or self.use_rebuild_expected_feat:
            # Variables pre-stored for Inference
            self.action_clusters = ActionClusters(self.num_classes, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])


            self.per_action_diff_list = {}
            self.first_action_diff_list = []
            self.per_action_diff_mean_std_weight = {}
            self.action_diff_dict = defaultdict(list)


        self.potential_action_generation_filtration_rate = rebuild_model_cfg['potential_action_generation_filtration_rate']

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def generate_expected_features(self, fpn_feats, error_desc_feats, action_cluster_centers, fpn_masks):
        # fpn_feats: List[B, C, T_i]
        # error_desc_feats: List[B, C, T_e]
        # action_cluster_centers: List[B, C, T_e]
        # fpn_masks: List[B, 1, T_i]

        diversity_mode = self.rebuild_model_cfg['diversity_mode']

        fpn_expected_feats = []
        fpn_expected_probs = []
        for i in range(len(fpn_feats)):
            if i > 0:
                fpn_expected_feats.append(fpn_feats[i])
                fpn_expected_probs.append(None)
                continue
            scale_factor = 2 ** i
            
            if diversity_mode == 'cluster_center':
                fpn_action_cluster_centers = action_cluster_centers[:, :, ::scale_factor]
                result = self.expected_feat_predictor(fpn_feats[i], fpn_action_cluster_centers, fpn_masks[i])
            else:
                fpn_error_desc_feats = error_desc_feats[:, :, ::scale_factor]
                result = self.expected_feat_predictor(fpn_feats[i], fpn_error_desc_feats, fpn_masks[i])
            
            if isinstance(result, tuple):
                expected_probs, expected_feat = result
                fpn_expected_probs.append(expected_probs)
            else:
                expected_feat = result
                fpn_expected_probs.append(None)
            
            fpn_expected_feats.append(expected_feat)
        
        return fpn_expected_feats, fpn_expected_probs

    
    def transform_feat_loss(self, transformed_feats, labels, margin=10.0):
        n = transformed_feats.size(0)
        losses = []

        # Calculate pairwise distances (L2 distance)
        # dists = ((transformed_feats.unsqueeze(1) - transformed_feats.unsqueeze(0)) ** 2).sum(dim=2)
        dists = calculate_distance(transformed_feats, transformed_feats, pairwise=True, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])

        for i in range(n):
            # Get distances of sample i to all others
            dist_i = dists[i]
            
            # Get same class and different class indices
            positive_indices = (labels == labels[i]) & (torch.arange(n) != i).to(self.device)
            negative_indices = (labels != labels[i])

            if positive_indices.sum() == 0 or negative_indices.sum() == 0:
                continue

            # Calculate minimum positive distance and maximum negative distance
            pos_dist = torch.min(dist_i[positive_indices])
            neg_dist = torch.max(dist_i[negative_indices])

            # Compute triplet loss
            triplet_loss = torch.relu(pos_dist - neg_dist + margin)
            losses.append(triplet_loss)

        if losses:
            final_loss = torch.mean(torch.stack(losses))
        else:
            final_loss = torch.tensor(0.0).to(self.device)
        
        return final_loss

    def forward(self, video_list, mode = 'none', weight=1, threshold=0.5):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)

        batched_bboxes, batched_bbox_classes, \
            batched_edge_maps, batched_inputs, batched_masks = self.preprocessing_gcn(video_list)

        feats, masks = self.backbone(batched_bboxes, batched_bbox_classes,
                                    batched_edge_maps, batched_inputs, batched_masks)

        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits, intermediate_feats = self.cls_head(fpn_feats, fpn_masks, output_intermediate=True)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        
        
        if not self.training and mode == 'none':
            results = self.inference(video_list, points, fpn_masks, out_cls_logits, out_offsets)
            return results
        
        results_frame = None
        fpn_feats_potential_sequences = None
        fpn_masks_potential_sequences = None
        batch_pred_error_desc_feats = None
        batch_segments_potential_sequences = None
        batch_labels_potential_sequences = None
        precise_optimization = None
        fpn_expected_probs = None
        if self.use_rebuild_expected_feat and 'cluster_transformed_feat' not in mode and 'accumulate_transformed_feat' not in mode:
            # extract prediction label per point/frame
            results = self.inference(video_list, points, fpn_masks, out_cls_logits, out_offsets)

            max_added_seq = self.rebuild_model_cfg['diversity_in_training'] if self.training else self.rebuild_model_cfg['diversity_max_added_seq']

            fpn_feats_potential_sequences, fpn_masks_potential_sequences, batch_pred_error_desc_feats, batch_pred_action_cluster_centers,\
                batch_segments_potential_sequences, batch_labels_potential_sequences, precise_optimization, results_frame = \
                    self.build_input_sequences(fpn_feats, fpn_masks, video_list, results, max_added_seq=max_added_seq)
            
            fpn_expected_feats, fpn_expected_probs = self.generate_expected_features(fpn_feats_potential_sequences, batch_pred_error_desc_feats, batch_pred_action_cluster_centers, fpn_masks_potential_sequences)
        else:
            fpn_expected_feats = None


        # ---------------- transform model's training & Cluster accumulation ----------------
        if self.use_transform_feat or 'accumulate_transformed_feat' in mode:
            results = self.inference(video_list, points, fpn_masks, out_cls_logits, out_offsets)
            batch_labels = []
            batch_transformed_feats = []



            if 'cluster_accumulate_method_gt' in mode:
                ## cluster_init_method: gt
                for idx, single_video_gt in enumerate(video_list):
                    gt_segments = single_video_gt['segments']
                    gt_labels = single_video_gt['labels']
                    transformed_feats = []
                    for segment_idx, (start, end) in enumerate(gt_segments):
                        action_feat = fpn_feats[0][idx, :, int(start):int(end)+1].detach()
                        action_feat = transform_frame_feat_to_action_feat(action_feat)
                        transformed_feat = self.transform_model(action_feat.unsqueeze(0)).squeeze(0)
                        transformed_feats.append(transformed_feat)
                    transformed_feats = torch.stack(transformed_feats, dim=0)
                    batch_labels.append(gt_labels)
                    batch_transformed_feats.append(transformed_feats)         
            else:
                ## cluster_init_method: pred_segment_gt_label
                for idx, (single_result, single_video_gt) in enumerate(zip(results, video_list)):
                    filtered_segments, filtered_labels = filter_segments_and_labels(single_video_gt, single_result,  
                                                                                    intersection_threshold=self.rebuild_model_cfg['intersection_threshold'])
                    if len(filtered_labels) > 0:
                        feat = fpn_feats[0][idx].detach()
                        transformed_feats = []
                        for start, end in filtered_segments:
                            action_feat = feat[:, start:end+1]
                            action_feat = transform_frame_feat_to_action_feat(action_feat)
                            transformed_feat = self.transform_model(action_feat.unsqueeze(0)).squeeze(0)
                            transformed_feats.append(transformed_feat)
                        
                        transformed_feats = torch.stack(transformed_feats, dim=0)
                        filtered_labels = torch.tensor(filtered_labels)
                        batch_labels.append(filtered_labels)
                        batch_transformed_feats.append(transformed_feats)
            





            if len(batch_labels) > 0:
                batch_labels = torch.cat(batch_labels, dim=0).to(self.device)
                batch_transformed_feats = torch.cat(batch_transformed_feats, dim=0).to(self.device)
                
                if self.use_transform_feat and self.training:
                    transform_loss = self.transform_feat_loss(batch_transformed_feats, batch_labels)

                    logging.info('transform_loss', transform_loss)

                    return {'transform_loss': transform_loss,
                            'final_loss' : transform_loss}

                self.action_clusters.add_feats(batch_transformed_feats, batch_labels)


        if 'cluster_transformed_feat' in mode and self.use_rebuild_expected_feat:
            self.action_clusters.update_clusters(self.rebuild_model_cfg['cluster_center_update_momentum'])
            # self.action_clusters.calculate_distances_and_weights()
            # for cls, radius in self.action_clusters.clusters_radius.items():
            #     print(f'Cluster {cls} Radius: {radius}')
            # print(self.action_clusters.centers_distances)
            # self.action_clusters.visualize_clusters()
            # self.action_clusters.visualize_distances_heatmap()
            # assert 0


        # ---------------- transform model's training & Cluster accumulation ----------------






        # return loss during training
        if self.training:
        

            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]


            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)
            # compute the loss and return

            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                fpn_feats_potential_sequences, fpn_expected_feats, batch_segments_potential_sequences, batch_labels_potential_sequences, precise_optimization,
                fpn_expected_probs, results_frame, video_list
            )

            return losses

        else:
            assert self.use_rebuild_expected_feat, 'Rebuild expected feature is required during inference'


            # MODE: Calculate Feat Difference
            if 'calculate_feature_difference' in mode:
                max_added_seq = self.rebuild_model_cfg['diversity_max_added_seq']
                inference_example_choice = self.rebuild_model_cfg['inference_example_choice']

                for sample_idx, (result, single_video_list) in enumerate(zip(results_frame, video_list)):
                    if inference_example_choice == 'gt':
                        # Accumulate threshold conditions using GT Segments
                        length = int(single_video_list['segments'][-1][1]) + 1
                        gt_segments = single_video_list['segments']
                        gt_labels = single_video_list['labels']
                        
                        for segment_idx, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
                            start = int(segment[0])
                            end = int(segment[1])
                            end = min(end, length - 1)
                            if start >= end:
                                continue

                            gt_labels_per_frame = to_frame_wise(single_video_list['segments'], single_video_list['labels'], None, length)
                            action_id = int(gt_labels_per_frame[start])

                            expected_action_feat = fpn_expected_feats[0][sample_idx * (1 + max_added_seq), :, start]
                            af_segment_feat = fpn_feats_potential_sequences[0][sample_idx * (1 + max_added_seq), :, start:end+1]
                            af_action_feat = transform_frame_feat_to_action_feat(af_segment_feat)
                            transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0)).squeeze(0)

                            if self.rebuild_loss_choice == 'mse':
                                expected_residual = expected_action_feat - self.action_clusters.clusters_centers[int(label)]
                                actual_residual = transformed_af_action_feat - self.action_clusters.clusters_centers[int(label)]
                                diff = calculate_distance(expected_residual, actual_residual, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                                # diff = calculate_distance(expected_action_feat, transformed_af_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            elif self.rebuild_loss_choice == 'residual':
                                residual_gt = transformed_af_action_feat - self.action_clusters.clusters_centers[action_id]
                                diff = calculate_distance(residual_gt, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            
                            if segment_idx == 0:
                                self.first_action_diff_list.append(diff)
                            elif action_id not in self.per_action_diff_list:
                                self.per_action_diff_list[action_id] = [diff]
                            else:
                                self.per_action_diff_list[action_id].append(diff)


                    else:
                        # Below are the ones containing predicted segments
                        segments = result['segments']
                        labels = result['labels']
                        length = int(video_list[sample_idx]['segments'][-1][1]) + 1
                        for segment_idx, (segment, label) in enumerate(zip(segments, labels)):
                            start = int(segment[0])
                            end = int(segment[1])
                            end = min(end, length - 1)
                            if start >= end:
                                continue
                            
                            if inference_example_choice == 'pred_segment_diversity_label':
                                # Calculate diff for each possibility and select the minimum diff
                                min_diff = 1e9
                                min_label = -1
                                for seq_idx in range(1 + max_added_seq):
                                    temp_label = batch_labels_potential_sequences[sample_idx * (1 + max_added_seq) + seq_idx][segment_idx]
                                    expected_action_feat = fpn_expected_feats[0][sample_idx * (1 + max_added_seq) + seq_idx, :, start]
                                    af_segment_feat = fpn_feats_potential_sequences[0][sample_idx * (1 + max_added_seq) + seq_idx, :, start:end+1]
                                    af_action_feat = transform_frame_feat_to_action_feat(af_segment_feat)
                                    transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0)).squeeze(0)
                
                                    if self.rebuild_loss_choice == 'mse':
                                        expected_residual = expected_action_feat - self.action_clusters.clusters_centers[int(label)]
                                        actual_residual = transformed_af_action_feat - self.action_clusters.clusters_centers[int(label)]
                                        diff = calculate_distance(expected_residual, actual_residual, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                                        # diff = calculate_distance(expected_action_feat, transformed_af_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                                    elif self.rebuild_loss_choice == 'residual':
                                        residual_gt = transformed_af_action_feat - self.action_clusters.clusters_centers[int(label)]
                                        diff = calculate_distance(residual_gt, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])

                                    if diff < min_diff:
                                        min_diff = diff
                                        min_label = temp_label

                                action_id = int(min_label)
                            elif inference_example_choice == 'pred_segment_gt_label':
                                gt_labels_per_frame = to_frame_wise(single_video_list['segments'], single_video_list['labels'], None, length)
                                gt_labels_in_segment = gt_labels_per_frame[start:end+1]
                                action_id = int(most_common_element(gt_labels_in_segment.tolist()))
                                expected_action_feat = fpn_expected_feats[0][sample_idx * (1 + max_added_seq), :, start]
                                af_segment_feat = fpn_feats_potential_sequences[0][sample_idx * (1 + max_added_seq), :, start:end+1]
                                af_action_feat = transform_frame_feat_to_action_feat(af_segment_feat)
                                transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0)).squeeze(0)
                                if self.rebuild_loss_choice == 'mse':
                                    expected_residual = expected_action_feat - self.action_clusters.clusters_centers[int(label)]
                                    actual_residual = transformed_af_action_feat - self.action_clusters.clusters_centers[int(label)]
                                    diff = calculate_distance(expected_residual, actual_residual, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                                    # diff = calculate_distance(expected_action_feat, transformed_af_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                                elif self.rebuild_loss_choice == 'residual':
                                    residual_gt = transformed_af_action_feat - self.action_clusters.clusters_centers[int(action_id)]
                                    diff = calculate_distance(residual_gt, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            else:
                                raise ValueError(f'Inference example choice {inference_example_choice} not supported')


                            if segment_idx == 0:
                                self.first_action_diff_list.append(diff)
                            elif action_id not in self.per_action_diff_list:
                                self.per_action_diff_list[action_id] = [diff]
                            else:
                                self.per_action_diff_list[action_id].append(diff)
                             
            if 'calculate_mean_std' in mode:
                quantiles = self.error_detection_inference_cfg['quantile']  # Extract quantile as hyperparameter
                if not isinstance(quantiles, list):
                    quantiles = [quantiles]  # If not a list, convert to list

                if self.first_action_diff_list:  # Ensure list is not empty
                    first_action_diff_std = torch.std(torch.stack(self.first_action_diff_list))
                    
                    for quantile in quantiles:
                        # Use torch.quantile to calculate specified quantile
                        first_action_diff_quantile = torch.quantile(torch.stack(self.first_action_diff_list), quantile)
                        logging.info(f'First Action, {quantile*100:.0f}th Percentile: {first_action_diff_quantile}, Std: {first_action_diff_std}')
                        if 'first' not in self.per_action_diff_mean_std_weight:
                            self.per_action_diff_mean_std_weight['first'] = {quantile: [(first_action_diff_quantile, first_action_diff_std, weight)]}
                        else:
                            if quantile not in self.per_action_diff_mean_std_weight['first']:
                                self.per_action_diff_mean_std_weight['first'][quantile] = [(first_action_diff_quantile, first_action_diff_std, weight)]
                            else:
                                self.per_action_diff_mean_std_weight['first'][quantile].append((first_action_diff_quantile, first_action_diff_std, weight))
                    self.first_action_diff_list.clear()


                for action_id in sorted(self.per_action_diff_list.keys()):
                    action_diff_list = self.per_action_diff_list[action_id]
                    diff_std = torch.std(torch.stack(action_diff_list))

                    for quantile in quantiles:
                        # Use torch.quantile to calculate specified quantile
                        diff_quantile = torch.quantile(torch.stack(action_diff_list), quantile)
                        logging.info(f'Action ID: {action_id}, {quantile*100:.0f}th Percentile: {diff_quantile}, Std: {diff_std}')
                        if action_id not in self.per_action_diff_mean_std_weight:
                            self.per_action_diff_mean_std_weight[action_id] = {quantile: [(diff_quantile, diff_std, weight)]}
                        else:
                            if quantile not in self.per_action_diff_mean_std_weight[action_id]:
                                self.per_action_diff_mean_std_weight[action_id][quantile] = [(diff_quantile, diff_std, weight)]
                            else:
                                self.per_action_diff_mean_std_weight[action_id][quantile].append((diff_quantile, diff_std, weight))
                self.per_action_diff_list.clear()

            if 'build_final_mean_std' in mode:
                logging.info('Building Final Mean and Std')
                self.per_action_threshold = {}
                for action_id in self.per_action_diff_mean_std_weight.keys():
                    self.per_action_threshold[action_id] = {}
                    for quantile, weight_list in self.per_action_diff_mean_std_weight[action_id].items():
                        total_weight = sum([w for _, _, w in weight_list])
                        final_mean = sum([mean * weight for mean, _, weight in weight_list]) / total_weight
                        valid_total_weight = sum(weight for _, std, weight in weight_list if not math.isnan(std))
                        if valid_total_weight > 0:
                            final_std = sum([weight * (std**2 + (mean - final_mean)**2) for mean, std, weight in weight_list if not math.isnan(std)]) / valid_total_weight
                            final_std = final_std**0.5
                        else:
                            final_std = 0.1
                        self.per_action_threshold[action_id][quantile] = (final_mean, final_std)
                        if action_id == 'first':
                            logging.info(f'First Action, Quantile: {quantile}, Mean: {final_mean}, Std: {final_std}')
                        else:
                            logging.info(f'Action ID: {action_id}, Quantile: {quantile}, Mean: {final_mean}, Std: {final_std}')

                # Record all missing threshold actions
                missing_threshold_actions = []
                for action_id in range(self.action_clusters.num_classes):
                    if action_id not in self.per_action_threshold:
                        missing_threshold_actions.append(action_id)

                # If there is a background threshold, use the background threshold directly
                if 0 in self.per_action_threshold:
                    for action_id in missing_threshold_actions:
                        self.per_action_threshold[action_id] = deepcopy(self.per_action_threshold[0])
                        logging.warning(f'Action ID: {action_id} not found in per_action_threshold, using background threshold')
                else:
                    # Collect all valid thresholds
                    valid_thresholds = []
                    for action_id in range(self.action_clusters.num_classes):
                        if action_id in self.per_action_threshold:
                            valid_thresholds.append(self.per_action_threshold[action_id])
                    
                    if valid_thresholds:
                        # Calculate mean threshold
                        mean_threshold = {}
                        for quantile in valid_thresholds[0].keys():
                            means = [t[quantile][0] for t in valid_thresholds]
                            stds = [t[quantile][1] for t in valid_thresholds]
                            mean_threshold[quantile] = (sum(means)/len(means), sum(stds)/len(stds))
                        
                        # Assign mean threshold to all missing actions
                        for action_id in missing_threshold_actions:
                            self.per_action_threshold[action_id] = mean_threshold
                            logging.warning(f'Action ID: {action_id} not found in per_action_threshold, using mean threshold')
                    else:
                        # If there are no valid thresholds, set default value 1 for all missing actions
                        default_threshold = {quantile: (1.0, 0.1) for quantile in self.error_detection_inference_cfg.get('quantile', [0.8])}
                        for action_id in missing_threshold_actions:
                            self.per_action_threshold[action_id] = default_threshold
                            logging.warning(f'[WARNING WARNING WARNING] Action ID: {action_id} not found in per_action_threshold, using default threshold 1.0')
                        raise ValueError(f'No valid threshold found for actions {missing_threshold_actions}')


            # New mode: Add testset samples to clusters
            if 'add_testset_to_clusters' in mode:
                for sample_idx, (result, single_video_list) in enumerate(zip(results_frame, video_list)):
                    # segments = result['segments']
                    # labels = result['labels']
                    # labels_error = result['labels_error']

                    segments = single_video_list['segments']
                    labels = single_video_list['labels']
                    labels_error = single_video_list['labels_error']
                    for segment, label, label_error in zip(segments, labels, labels_error):
                        start = int(segment[0])
                        end = int(segment[1])

                        af_segment_feat = fpn_feats[0][sample_idx, :, start:end+1]
                        af_action_feat = transform_frame_feat_to_action_feat(af_segment_feat)
                        transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0))

                        action_id = int(label)

                        self.action_clusters.add_test_feats(transformed_af_action_feat, torch.tensor([action_id]), is_error=(label_error > 0))


            if 'inference' in mode:
                max_added_seq = self.rebuild_model_cfg['diversity_max_added_seq']
                scale_choice = self.rebuild_model_cfg['scale_choice']
                scale_factors = {}
                if scale_choice is not None:
                    # Add scale factor for first action
                    mean, std = self.per_action_threshold['first'][scale_choice]
                    scale_factors['first'] = mean
                    # Add scale factors for other actions
                    for label in range(self.action_clusters.num_classes):
                        mean, std = self.per_action_threshold[label][scale_choice]
                        scale_factors[label] = mean


                for sample_idx, result in enumerate(results_frame):
                    segments = result['segments']
                    labels = result['labels']
                    original_labels = deepcopy(labels)
                    corrected_labels = deepcopy(labels)

                    diff_and_threshold = {quantile: [] for quantile in self.per_action_threshold[0].keys()}
                    all_diff_and_threshold = {quantile: [] for quantile in self.per_action_threshold[0].keys()}
                    quantile_labels = {quantile: deepcopy(labels) for quantile in self.per_action_threshold[0].keys()}

                    for segment_idx, (segment, label) in enumerate(zip(segments, labels)):
                        # if segment_idx == 0:
                        #     # For the first segment, directly add a representation indicating correctness
                        #     for quantile in self.per_action_threshold[0].keys():
                        #         diff_and_threshold[quantile].append((0.0, 0.0, False))
                        #     continue

                        start = int(segment[0])
                        end = int(segment[1])
                        end = min(end, int(result['end_frame']))
                        if end <= start:
                            for quantile in self.per_action_threshold[0].keys():
                                diff_and_threshold[quantile].append((0.0, 0.0, False))
                                all_diff_and_threshold[quantile].append([])
                            continue
                        if end == int(result['end_frame']) - 1:
                            end = int(result['end_frame'])

        
                        min_diff = torch.tensor(1e9)
                        min_label = -1

                        af_segment_feat = fpn_feats_potential_sequences[0][sample_idx * (1 + max_added_seq), :, start:end+1] 
                        af_action_feat = transform_frame_feat_to_action_feat(af_segment_feat)
                        transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0)).squeeze(0)

                        pred_label = batch_labels_potential_sequences[sample_idx * (1 + max_added_seq)][segment_idx]

                        # Store all differences and thresholds for each quantile
                        all_diffs = {quantile: [] for quantile in self.per_action_threshold[0].keys()}

                        for seq_idx in range(1 + max_added_seq):
                            temp_label = batch_labels_potential_sequences[sample_idx * (1 + max_added_seq) + seq_idx][segment_idx]
                            if pred_label == 0 and seq_idx > 0 and self.rebuild_model_cfg['separate_action_and_bg']:
                                break
                            if pred_label != 0 and temp_label == 0 and self.rebuild_model_cfg['separate_action_and_bg']:
                                continue
                            expected_action_feat = fpn_expected_feats[0][sample_idx * (1 + max_added_seq) + seq_idx, :, start]

                            if self.rebuild_loss_choice == 'mse':
                                expected_residual = expected_action_feat - self.action_clusters.clusters_centers[int(label)]
                                actual_residual = transformed_af_action_feat - self.action_clusters.clusters_centers[int(label)]
                                diff = calculate_distance(expected_residual, actual_residual, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                                # diff = calculate_distance(expected_action_feat, transformed_af_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            elif self.rebuild_loss_choice == 'residual':
                                residual_gt = transformed_af_action_feat - self.action_clusters.clusters_centers[int(temp_label)]
                                diff = calculate_distance(residual_gt, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            
                            if scale_choice is not None:
                                # Apply the scale factor to the first action
                                if segment_idx == 0:
                                    diff = diff / scale_factors['first']
                                else:
                                    diff = diff / scale_factors[temp_label]

                            # Record the difference and threshold for each quantile for the current label
                            threshold_key = 'first' if segment_idx == 0 else temp_label
                            for quantile, (mean, std) in self.per_action_threshold[threshold_key].items():
                                if scale_choice is not None:
                                    if segment_idx == 0:
                                        quantile_threshold = (mean + threshold * std) / scale_factors['first']
                                    else:
                                        quantile_threshold = (mean + threshold * std) / scale_factors[temp_label]
                                else:
                                    quantile_threshold = mean + threshold * std
                                all_diffs[quantile].append((diff.detach().cpu().numpy(), quantile_threshold.detach().cpu().numpy(), temp_label))

                            if diff < min_diff:
                                min_diff = diff
                                min_label = temp_label            
                            logging.info(f'diff: {diff}, min_diff: {min_diff}, min_label: {min_label}, temp_label: {temp_label}')
                            logging.info(f'start: {start}, end: {end}')

                        action_id = int(min_label)
                        if action_id == -1:
                            action_id = 0
                        corrected_labels[segment_idx] = action_id

                        # Record the threshold corresponding to the minimum difference
                        threshold_key = 'first' if segment_idx == 0 else action_id
                        for quantile, (mean, std) in self.per_action_threshold[threshold_key].items():
                            if scale_choice is not None:
                                if segment_idx == 0:
                                    quantile_threshold = (mean + threshold * std) / scale_factors['first']
                                else:
                                    quantile_threshold = (mean + threshold * std) / scale_factors[action_id]
                            else:
                                quantile_threshold = mean + threshold * std
                            if min_diff > quantile_threshold:
                                quantile_labels[quantile][segment_idx] = -1
                            diff_and_threshold[quantile].append((min_diff.detach().cpu().numpy(), quantile_threshold.detach().cpu().numpy(), (min_diff > quantile_threshold).detach().cpu().numpy()))
                            all_diff_and_threshold[quantile].append(all_diffs[quantile])

                    result['diff_and_threshold'] = diff_and_threshold
                    result['all_diff_and_threshold'] = all_diff_and_threshold
                    result['original_labels'] = original_labels
                    result['corrected_labels'] = corrected_labels
                    result['quantile_labels'] = quantile_labels
                return results_frame

    @torch.no_grad()
    def preprocessing_gcn(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        device = self.device
       
        feats = [x['feats'] for x in video_list]  # List of [C, T]
        bboxes = [x['bbox'].permute(1, 2, 0) for x in video_list] if 'bbox' in video_list[0] else None  # List of [num_node, 4, T]
        bbox_classes = [x['bbox_class'].permute(1, 0) for x in video_list] if 'bbox_class' in video_list[0] else None  # List of [num_node, T]
        edge_maps = [x['edge_map'].permute(1, 2, 0) for x in video_list] if 'edge_map' in video_list[0] else None  # List of [num_node, num_node, T]

        feats_lens = torch.tensor([feat.shape[-1] for feat in feats], device=device)  # [B]
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            # an empty batch with padding
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            # refill the batch
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            
            # batch bbox shape B, num_node, 4, T
            if bboxes is not None:
                batch_bboxes_shape = [len(bboxes), bboxes[0].shape[0], bboxes[0].shape[1], max_len]
                batched_bboxes = bboxes[0].new_full(batch_bboxes_shape, padding_val)
                for bbox, pad_bbox in zip(bboxes, batched_bboxes):
                    pad_bbox[..., :bbox.shape[-1]].copy_(bbox)
            else:
                batched_bboxes = None
            
            # batch bbox_class shape B, num_node, T
            if bbox_classes is not None:
                batch_bbox_classes_shape = [len(bbox_classes), bbox_classes[0].shape[0], max_len]
                batched_bbox_classes = bbox_classes[0].new_full(batch_bbox_classes_shape, padding_val)
                for bbox_class, pad_bbox_class in zip(bbox_classes, batched_bbox_classes):
                    pad_bbox_class[..., :bbox_class.shape[-1]].copy_(bbox_class)
            else:
                batched_bbox_classes = None
            
            # batch edge map shape B, num_node, num_node, T
            if edge_maps is not None:
                batch_edge_maps_shape = [len(edge_maps), edge_maps[0].shape[0], edge_maps[0].shape[1], max_len]
                batched_edge_maps = edge_maps[0].new_full(batch_edge_maps_shape, padding_val)
                for edge_map, pad_edge_map in zip(edge_maps, batched_edge_maps):
                    pad_edge_map[..., :edge_map.shape[-1]].copy_(edge_map)
            else:
                batched_edge_maps = None

            
            # padding error_descriptions_feat
            for sample in video_list:
                if 'error_descriptions_feat' in sample:
                    error_descriptions_feat = sample['error_descriptions_feat']
                    T = error_descriptions_feat.shape[0]
                    if T < max_len:
                        padding_size = (0, 0, 0, max_len - T)  # Pad to the first dimension, not affecting the second dimension
                        sample['error_descriptions_feat'] = F.pad(
                            error_descriptions_feat, padding_size, 'constant', value=padding_val
                        )
                
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            if bboxes is not None:  
                padding_bboxes_size = [0, max_len - bboxes[0].shape[-1]]
            else:
                padding_bboxes_size = None
            if bbox_classes is not None:
                padding_bbox_classes_size = [0, max_len - bboxes[0].shape[-1]]
            else:
                padding_bbox_classes_size = None
            if edge_maps is not None:
                padding_edge_maps_size = [0, max_len - bboxes[0].shape[-1]]
            else:
                padding_edge_maps_size = None
                
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)
            if bboxes is not None:
                batched_bboxes = F.pad(
                    bboxes[0], padding_bboxes_size, value=padding_val).unsqueeze(0)
            else:
                batched_bboxes = None

            if bbox_classes is not None:
                batched_bbox_classes = F.pad(
                    bbox_classes[0], padding_bbox_classes_size, value=padding_val).unsqueeze(0)
            else:
                batched_bbox_classes = None

            if edge_maps is not None:
                batched_edge_maps = F.pad(
                    edge_maps[0], padding_edge_maps_size, value=padding_val).unsqueeze(0)
            else:
                batched_edge_maps = None

            
            # padding error_descriptions_feat
            if 'error_descriptions_feat' in video_list[0]:
                error_descriptions_feat = video_list[0]['error_descriptions_feat']
                T = error_descriptions_feat.shape[0]
                if T < max_len:
                    padding_size = (0, 0, 0, max_len - T)
                    video_list[0]['error_descriptions_feat'] = F.pad(
                        error_descriptions_feat, padding_size, 'constant', value=padding_val
                    )

        # generate the mask
        batched_masks = torch.arange(max_len, device=feats_lens.device)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        if batched_bboxes is not None:
            batched_bboxes = batched_bboxes.to(self.device)
        if batched_bbox_classes is not None:
            batched_bbox_classes = batched_bbox_classes.to(self.device)
        if batched_edge_maps is not None:
            batched_edge_maps = batched_edge_maps.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_bboxes, batched_bbox_classes, batched_edge_maps, batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset


    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets


    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets,
        fpn_feats=None, fpn_expected_feats=None, segments=None, labels=None, precise_optimization=None,
        fpn_expected_probs=None, results_frame=None, video_list=None
    ):     

        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        # fpn_feats, fpn_expected_feats: Size: List [B, C, T_i]
        # segments: List[T x 2] with list length = batchsize
        # labels: List[T] with list length = batchsize
        # precise_optimization: List[bool] with list length = batchsize
        # fpn_expected_probs: List[B, C, T_i]
        valid_mask = torch.cat(fpn_masks, dim=1)
        
        # Calculate expected feature loss
        expected_feat_loss = torch.tensor(0.0).to(self.device)
        predict_next_action_label_loss = torch.tensor(0.0).to(self.device)
        action_feat_to_center_distance_loss = torch.tensor(0.0).to(self.device)
        if self.use_rebuild_expected_feat:
            assert all(item is not None for item in [fpn_feats, fpn_expected_feats, segments, labels, precise_optimization])
            expected_feat_loss = self.rebuild_expected_feat_loss(fpn_feats, fpn_expected_feats, segments, labels, precise_optimization)
            if self.predict_class:
                predict_next_action_label_loss = self.predict_next_action_label_loss(fpn_expected_probs, results_frame, video_list)
            if self.rebuild_model_cfg['action_feat_to_center_distance_loss_dist'] > 0:
                action_feat_to_center_distance_loss = self.action_feat_to_center_distance_loss(fpn_feats, results_frame, video_list, self.rebuild_model_cfg['action_feat_to_center_distance_loss_dist'])

        # Classification loss
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)
        cls_loss = sigmoid_focal_loss(torch.cat(out_cls_logits, dim=1)[valid_mask], gt_target, reduction='sum') / self.loss_normalizer

        # Regression loss
        reg_loss = ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction='sum') / self.loss_normalizer if num_pos > 0 else 0 * pred_offsets.sum()
        loss_weight = self.train_loss_weight if self.train_loss_weight > 0 else cls_loss.detach() / max(reg_loss.item(), 0.01)
    
        if self.predict_class:
            final_loss = predict_next_action_label_loss
        else:
            final_loss = cls_loss * self.cls_loss_weight + reg_loss * loss_weight + expected_feat_loss + action_feat_to_center_distance_loss
        
        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'expected_feat_loss': expected_feat_loss,
            'action_feat_to_center_distance_loss': action_feat_to_center_distance_loss,
            'final_loss': final_loss
        }
        if self.predict_class:
            loss_dict['predict_next_action_label_loss'] = predict_next_action_label_loss

        return loss_dict

    def rebuild_expected_feat_loss(self, fpn_feats, fpn_expected_feats, segments, labels, precise_optimization):
        '''
        fpn_feats, fpn_expected_feats. List, Tensor Size: [B, C, T_i]
        segments: List[T x 2] with list length = batchsize
        labels: List[T] with list length = batchsize
        precise_optimization: List[bool] with list length = batchsize
        '''
        diversity_in_training = self.rebuild_model_cfg['diversity_in_training']
        losses = []
        positive_losses = []
        negative_losses = []
        for bs_idx in range(0, len(fpn_feats[0]), diversity_in_training + 1):
            single_gt_feats = fpn_feats[0][bs_idx]
            single_segments = segments[bs_idx]
            
            segment_lengths = [len(segments[i]) for i in range(bs_idx, min(bs_idx+diversity_in_training+1, len(segments)))]
            assert all(length == segment_lengths[0] for length in segment_lengths), f"Segments lengths from bs_idx {bs_idx} to {min(bs_idx+diversity_in_training, len(segments)-1)} are not equal: {segment_lengths}\nsegments: {segments}"
            label_lengths = [len(labels[i]) for i in range(bs_idx, min(bs_idx+diversity_in_training+1, len(labels)))]
            assert all(length == label_lengths[0] for length in label_lengths), f"Labels lengths from bs_idx {bs_idx} to {min(bs_idx+diversity_in_training, len(labels)-1)} are not equal: {label_lengths}\nlabels: {labels}"
            
            for seg_idx, segment in enumerate(single_segments):
                start = int(segment[0])
                end = int(segment[1])
                
                feat = single_gt_feats[:, start:end+1]
                
                af_action_feat = transform_frame_feat_to_action_feat(feat) # Size: [C]
                transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0)).squeeze(0) # Size: [C]

                segment_loss_pos = torch.tensor(0.0).to(self.device)
                segment_loss_neg = torch.tensor(1.0).to(self.device)
                for sample_idx in range(diversity_in_training + 1):
                    single_precise_optimization = precise_optimization[bs_idx + sample_idx]
                    single_expected_feats = fpn_expected_feats[0][bs_idx + sample_idx]
                    expected_action_feat = single_expected_feats[:, start] # Size: [C]
                    label = labels[bs_idx + sample_idx][seg_idx]

                    cluster_center = self.action_clusters.clusters_centers[int(label)]
                    cluster_radius = self.action_clusters.clusters_radius[int(label)]

                    assert cluster_center is not None, f'Cluster center or radius is None for label {label}, cluster_center: {self.action_clusters.clusters_centers}, radius: {self.action_clusters.clusters_radius}'

                    if self.rebuild_loss_choice == 'mse':
                        if single_precise_optimization:
                            expected_residual = expected_action_feat - self.action_clusters.clusters_centers[int(label)]
                            actual_residual = transformed_af_action_feat - self.action_clusters.clusters_centers[int(label)]
                            dist= calculate_distance(expected_residual, actual_residual, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            # dist = calculate_distance(transformed_af_action_feat, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            segment_loss_pos += dist
                            positive_losses.append(dist)
                        else:
                            dist = calculate_distance(cluster_center, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                            segment_loss_neg += dist
                            negative_losses.append(dist)
                    elif self.rebuild_loss_choice == 'residual':
                        residual_gt = transformed_af_action_feat - cluster_center
                        dist = calculate_distance(residual_gt, expected_action_feat, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                        if single_precise_optimization:
                            segment_loss_pos += dist
                            positive_losses.append(dist)
                        else:
                            segment_loss_neg += dist
                            negative_losses.append(dist)
                segment_loss = segment_loss_pos / segment_loss_neg
                losses.append(segment_loss)
        
        if len(positive_losses) > 0:
            logging.info(f'Positive Losses: {torch.mean(torch.stack(positive_losses))}')
            
        if negative_losses:
            if len(negative_losses) > 0:
                logging.info(f'Negative Losses: {torch.mean(torch.stack(negative_losses))}')
        loss = torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(self.device)

        return loss

    def action_feat_to_center_distance_loss(self, fpn_feats, video_list, results, action_feat_to_center_distance_loss_dist=1):
        pull_losses = []
        push_losses = []
        for idx, (single_gt_feats, single_video_gt, single_result) in enumerate(zip(fpn_feats[0], video_list, results)):
            filtered_segments, filtered_labels = filter_segments_and_labels(single_video_gt, single_result, 
                                                                            intersection_threshold=self.rebuild_model_cfg['intersection_threshold'])
            
            for segment, label in zip(filtered_segments, filtered_labels):
                start = int(segment[0]) 
                end = int(segment[1])
                
                feat = single_gt_feats[:, start:end+1]
                
                af_action_feat = transform_frame_feat_to_action_feat(feat) # Size: [C]
                transformed_af_action_feat = self.transform_model(af_action_feat.unsqueeze(0)).squeeze(0) # Size: [C]
                
                # Calculate the distance between the action feature and the center of the same class
                same_class_center = self.action_clusters.clusters_centers[int(label)]
                same_class_dist = calculate_distance(transformed_af_action_feat, same_class_center, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                pull_loss = same_class_dist  # Use the distance as the loss, minimizing the loss means pulling closer
                pull_losses.append(pull_loss)
                
                # Calculate the distance between the action feature and the centers of other classes
                other_class_centers = [self.action_clusters.clusters_centers[i] for i in range(self.num_classes) if i != int(label)]
                
                for center in other_class_centers:
                    dist = calculate_distance(transformed_af_action_feat, center, use_norm=self.rebuild_model_cfg['use_norm'], distance_type=self.rebuild_model_cfg['distance_type'])
                    push_loss = -dist
                    push_losses.append(push_loss)  # Use negative distance as the loss, minimizing the loss means maximizing the distance
        
        pull_loss = torch.mean(torch.stack(pull_losses)) if pull_losses else torch.tensor(0.0).to(self.device)
        push_loss = torch.mean(torch.stack(push_losses)) if push_losses else torch.tensor(0.0).to(self.device)
        
        # loss = max((pull_loss + push_loss + action_feat_to_center_distance_loss_dist), torch.tensor(0.0).to(self.device))
        loss = pull_loss / (-push_loss + 1e-6)
        
        return loss



    def predict_next_action_label_loss(self, fpn_expected_probs, results_frame, video_list):
        expected_probs = fpn_expected_probs[0]
        total_loss = 0
        total_samples = 0
        for batch_idx, (single_expected_prob, single_results_frame, single_video_list) in enumerate(zip(expected_probs, results_frame, video_list)):
            gt_segments = single_video_list['segments']
            gt_labels = single_video_list['labels']
            pred_segments = single_results_frame['segments']
            try:
                length = single_video_list['end'] - single_video_list['start']
            except:
                logging.info(single_video_list)
                raise
            # Initialize a list to store the indices of valid predicted segments
            valid_pred_indices = []
            # Iterate over all predicted segments
            for pred_idx, pred_segment in enumerate(pred_segments):
                pred_start, pred_end = pred_segment
                pred_length = pred_end - pred_start

                # Check the intersection of each predicted segment with all true segments
                for gt_segment in gt_segments:
                    gt_start, gt_end = gt_segment
                    
                    # Calculate the intersection
                    intersection_start = max(pred_start, gt_start)
                    intersection_end = min(pred_end, gt_end)
                    intersection_length = max(0, intersection_end - intersection_start)
                    
                    # Calculate the ratio of the intersection to the predicted segment length
                    overlap_ratio = intersection_length / pred_length
                    
                    # If the ratio is greater than 80%, add the index of the predicted segment to valid_pred_indices
                    if overlap_ratio > 0.8:
                        valid_pred_indices.append(pred_idx)
                        break  # Once a true segment that meets the condition is found, no need to check other true segments
            
            
            gt_labels_frame = to_frame_wise(gt_segments, gt_labels, None, length)

            engaged_label_frame_idx = [int(pred_segments[idx][0]) for idx in valid_pred_indices]
            engaged_probs = [single_expected_prob[:, idx] for idx in engaged_label_frame_idx]
            engaged_gt_labels = [gt_labels_frame[idx] for idx in engaged_label_frame_idx]

            # Calculate the loss
            if engaged_probs and engaged_gt_labels:
                engaged_probs = torch.stack(engaged_probs)
                engaged_gt_labels = torch.tensor(engaged_gt_labels, dtype=torch.long, device=engaged_probs.device)
                loss = F.cross_entropy(engaged_probs, engaged_gt_labels, reduction='mean')
                total_loss += loss
                total_samples += len(engaged_gt_labels)

        if total_samples > 0:
            avg_loss = total_loss
        else:
            avg_loss = torch.tensor(0.0, device=total_loss.device)

        return avg_loss

            

            

        
    def bfs_find_next_actions(self, task_name, executed_sequence):
        """
        Use simple retrieval method to get all subsequent nodes in the task graph that have been executed
        
        Args:
            task_name: Task name
            executed_sequence: Executed action sequence
        
        Returns:
            prioritized_actions: List of possible next actions, prioritized
        """
        # Load the task graph
        if self.rebuild_model_cfg['dataset_name'] == 'HoloAssist':
            graphs = json.load(open('libs/datasets/holoassist_dag_buffer/task_graphs.json', 'r'))
        elif self.rebuild_model_cfg['dataset_name'] == 'CaptainCook4D':
            graphs = json.load(open('/home/weijin/source/MistakeDetection/FAFP/CaptainCook4D/CC4D_process/task_graph.json', 'r'))
        elif self.rebuild_model_cfg['dataset_name'] == 'EgoPER':
            if self.rebuild_model_cfg['use_trainset_graph']:
                graphs = json.load(open('libs/datasets/egoper_dag_buffer/task_graphs.json', 'r'))
            else:
                graphs = {
                    'tea': [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (0, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
                    'pinwheels': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)],
                    'oatmeal': [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
                    'quesadilla': [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8), (8, 9)],
                    'coffee': [(0, 1), (1, 2), (2, 13), (0, 5), (5, 13), (0, 6), (6, 7), (7, 8), (8, 12), (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 3), (3, 4), (4, 16)]
                }
        else:
            raise ValueError(f"Unknown dataset: {self.rebuild_model_cfg['dataset_name']}")

        # If the sequence is empty, return an empty set
        if not executed_sequence:
            return []

        # Build adjacency list
        adjacency_list = {}
        for start, end in graphs[task_name]:
            if start not in adjacency_list:
                adjacency_list[start] = []
            adjacency_list[start].append(end)

        # Get all possible next actions
        next_actions_priority = set()  # High-priority actions (direct successors)
        next_actions = set()          # All possible successors

        # Get the direct successors of the last executed action
        last_action = executed_sequence[-1]
        if last_action in adjacency_list:
            next_actions_priority.update(adjacency_list[last_action])

        # Get all successors of all executed actions
        for action in executed_sequence:
            if action in adjacency_list:
                next_actions.update(adjacency_list[action])

        # Remove executed actions
        executed_set = set(executed_sequence)
        next_actions_priority.difference_update(executed_set)
        next_actions.difference_update(executed_set)
        
        # Remove duplicates (high-priority actions should not appear in regular priority)
        next_actions.difference_update(next_actions_priority)

        # Create priority list
        prioritized_actions = list(next_actions_priority) + list(next_actions)

        # Handle special category (replace num_classes with 0)
        prioritized_actions = [0 if action == self.num_classes else action for action in prioritized_actions]

        return prioritized_actions

    def get_all_valid_next_actions(self, task_name, executed_sequence):
        def find_all_matching_subsequences(graphs, task_name, sequence):
            # Build adjacency list
            adjacency_list = {}
            for start, end in graphs[task_name]:
                if start not in adjacency_list:
                    adjacency_list[start] = []
                adjacency_list[start].append(end)

            # Initialize dynamic programming tables
            n = len(sequence)
            dp = [1] * n  
            # Modify initialization to ensure each position has the correct key-value pair
            subsequences = []
            for i in range(n):
                subsequences.append({})
                subsequences[i][1] = [[sequence[i]]]  # Use 1 as the initial length key

            # Fill dp table and collect subsequences
            for i in range(n):
                for j in range(i):
                    if sequence[j] in adjacency_list and sequence[i] in adjacency_list[sequence[j]]:
                        current_length = dp[j] + 1
                        if current_length > dp[i]:
                            # Found a longer subsequence
                            dp[i] = current_length
                            subsequences[i][current_length] = []
                            for prev_subseq in subsequences[j][dp[j]]:
                                new_subseq = prev_subseq + [sequence[i]]
                                if new_subseq not in subsequences[i][current_length]:
                                    subsequences[i][current_length].append(new_subseq)
                        elif current_length == dp[i]:
                            # Found another subsequence of same length
                            if current_length not in subsequences[i]:
                                subsequences[i][current_length] = []
                            for prev_subseq in subsequences[j][dp[j]]:
                                new_subseq = prev_subseq + [sequence[i]]
                                if new_subseq not in subsequences[i][current_length]:
                                    subsequences[i][current_length].append(new_subseq)

            # Collect all maximal subsequences
            max_length = max(dp)
            all_subsequences = []
            for i in range(n):
                if dp[i] == max_length and max_length in subsequences[i]:
                    for subseq in subsequences[i][max_length]:
                        if subseq not in all_subsequences:
                            all_subsequences.append(subseq)

            # Merge connected subsequences
            merged_subsequences = []
            for subseq in all_subsequences:
                merged = subseq[:]  # Create a copy
                for node in subseq:
                    if node in adjacency_list:
                        for neighbor in adjacency_list[node]:
                            for other_subseq in all_subsequences:
                                if neighbor in other_subseq:
                                    for n in other_subseq:
                                        if n not in merged:
                                            merged.append(n)
                if merged not in merged_subsequences:
                    merged_subsequences.append(merged)

            # Merge all nodes from subsequences
            all_nodes = set()
            for subseq in merged_subsequences:
                all_nodes.update(subseq)

            return all_nodes

        def validate_and_extend_sequence(graphs, task_name, sequence):
            # Find all matching subsequences that fit the graph
            matched_sequences = find_all_matching_subsequences(graphs, task_name, sequence)

            # Build reverse lookup for extending sequences
            reverse_lookup = {}
            for start, end in graphs[task_name]:
                if end not in reverse_lookup:
                    reverse_lookup[end] = []
                reverse_lookup[end].append(start)

            # Extend all sequences backwards
            extended_sequences = set(matched_sequences)
            while extended_sequences:
                current_start = min(extended_sequences)  # Use minimum node as starting point
                if current_start in reverse_lookup:
                    for pred in reverse_lookup[current_start]:
                        if pred not in extended_sequences:
                            extended_sequences.add(pred)
                            break
                break  # Only extend once to maintain sequence validity

            return list(extended_sequences)

        # -----------------------------------------------------
        
        # Define the graph structure for each dataset
        egoper_taskgraph = {
            'tea': [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (0, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
            'pinwheels': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)],
            'oatmeal': [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
            'quesadilla': [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8), (8, 9)],
            'coffee': [(0, 1), (1, 2), (2, 13), (0, 5), (5, 13), (0, 6), (6, 7), (7, 8), (8, 12), (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 3), (3, 4), (4, 16)]
        }

        
        if self.rebuild_model_cfg['dataset_name'] == 'HoloAssist':
            holoassist_trainset_graphs = json.load(open('libs/datasets/holoassist_dag_buffer/task_graphs.json', 'r'))
            graphs = holoassist_trainset_graphs
        elif self.rebuild_model_cfg['dataset_name'] == 'CaptainCook4D':
            cc4d_trainset_graphs = json.load(open('/home/weijin/source/MistakeDetection/FAFP/CaptainCook4D/CC4D_process/task_graph.json', 'r'))
            graphs = cc4d_trainset_graphs
        elif self.rebuild_model_cfg['dataset_name'] == 'EgoPER':
            if self.rebuild_model_cfg['use_trainset_graph']:
                egoper_trainset_graphs = json.load(open('libs/datasets/egoper_dag_buffer/task_graphs.json', 'r'))
                graphs = egoper_trainset_graphs
            else:
                graphs = egoper_taskgraph
        else:
            raise ValueError(f"Unknown dataset: {self.rebuild_model_cfg['dataset_name']}")

        # Clean the executed_sequence to remove noise
        cleaned_sequence = validate_and_extend_sequence(graphs, task_name, executed_sequence)

        # Collect all unique possible next actions from all nodes in the cleaned sequence
        adjacency_list = {}
        for start, end in graphs[task_name]:
            if start not in adjacency_list:
                adjacency_list[start] = []
            adjacency_list[start].append(end)

        next_actions_priority = set()
        if cleaned_sequence:
            # Prioritize the direct successors of the last valid action
            last_valid_action = cleaned_sequence[-1]
            if last_valid_action in adjacency_list:
                next_actions_priority.update(adjacency_list[last_valid_action])
        
        # Gather other possible actions
        possible_next_actions = set()
        for task in cleaned_sequence:
            if task in adjacency_list:
                possible_next_actions.update(adjacency_list[task])

        # Remove already executed actions and overlaps
        possible_next_actions.difference_update(cleaned_sequence)
        possible_next_actions.difference_update(next_actions_priority)  # Ensure no duplicates

        # Create a prioritized list with last valid actions' successors at the front
        prioritized_actions = list(next_actions_priority) + list(possible_next_actions)

        prioritized_actions = [0 if action == self.num_classes else action for action in prioritized_actions]

        return prioritized_actions



    def build_input_sequences(self, fpn_feats, fpn_masks, video_list, results, max_added_seq=3):
        assert len(video_list) == len(results) == len(fpn_feats[0]) == len(fpn_masks[0])
        # fpn_feats: List of tensors [B, C, T_i], where T_i is the number of frames in the ith layer of FPN
        # fpn_masks: List of tensors [B, T_i]
        use_multiple_possibilities = self.rebuild_model_cfg['use_multiple_possibilities']
        results_frame_type = self.rebuild_model_cfg['results_frame_type']
        intersection_threshold = self.rebuild_model_cfg['intersection_threshold']
        use_random_uncertainty = self.rebuild_model_cfg['random_uncertainty']



        insert_empty_label_flag = False


        
        def filter_actions(labels, segments, filtration_rate):
            # Calculate the duration of each action
            durations = [seg[1] - seg[0] + 1 for seg in segments]
            
            # Determine the filtering criteria (proportion or fixed value)
            if isinstance(filtration_rate, float):  # filtration_rate is a proportion
                total_frames = sum(durations)
                min_duration = total_frames * filtration_rate
            else:  # filtration_rate is a fixed number of frames
                min_duration = filtration_rate
            
            # Filter out actions with duration less than min_duration
            filtered_labels = []
            filtered_segments = []
            for label, segment, duration in zip(labels, segments, durations):
                if duration >= min_duration:
                    filtered_labels.append(label)
                    filtered_segments.append(segment)
            
            if len(filtered_labels) == 0:
                filtered_labels = [0]
                filtered_segments = None
            return filtered_labels, filtered_segments

        if self.rebuild_model_cfg['dataset_name'] == 'CaptainCook4D':
            action_id_to_str = None
        else:   
            action_id_to_str = video_list[0]['action_id_to_str']
        
        
        if 'error_descriptions_to_feat' in video_list[0]:
            error_descriptions_to_feat = video_list[0]['error_descriptions_to_feat']
        else:
            error_descriptions_to_feat = None

        results_frame = deepcopy(results)

        batch_pred_error_desc_feats = []
        batch_pred_action_cluster_centers = []
        precise_optimization = []
        batch_segments = []
        batch_labels = []

        for idx, (single_result, single_video_list) in enumerate(zip(results, video_list)):
            # Process each sample in the batch

            # === Predicted Sequence Processing ===
            # Using predicted segments and predicted labels (pred)
            segments = single_result['segments']  # Predicted segments
            labels = single_result['labels']      # Predicted labelss
            scores = single_result['scores']      # Predicted scores
            length = fpn_feats[0].shape[2]

            # Convert predicted segments and labels to frame-wise predictions
            pred = to_frame_wise(segments, labels, scores, length, fps=single_result['fps'])
            labels_frame, segments_frame = to_segments(pred)
            results_frame[idx]['labels'] = torch.tensor(labels_frame)
            results_frame[idx]['segments'] = torch.tensor(segments_frame)
            results_frame[idx]['end_frame'] = single_video_list['segments'][-1,1]


            labels_used_in_potential_actions = None
            segments_used_in_potential_actions = None

            # Build error description features for the predicted sequence
            if use_multiple_possibilities == 'single_pred' or use_multiple_possibilities == 'single' or (not self.training and not self.rebuild_model_cfg['debug_all_gt']):
                pred_error_desc_feats = []
                pred_action_cluster_centers = []
                for action_id in pred:
                    if action_id_to_str is not None:
                        error_desc = action_id_to_str[int(action_id)]
                    else:
                        error_desc = None
                    if error_descriptions_to_feat is not None:
                        error_desc_feat = error_descriptions_to_feat[error_desc]
                        pred_error_desc_feats.append(error_desc_feat)
                    pred_action_cluster_centers.append(self.action_clusters.clusters_centers[int(action_id)])
                pred_error_desc_feats = torch.cat(pred_error_desc_feats, dim=0).permute(1, 0)  if pred_error_desc_feats else None
                pred_action_cluster_centers = torch.stack(pred_action_cluster_centers, dim=0).permute(1, 0)
                batch_pred_error_desc_feats.append(pred_error_desc_feats)
                batch_pred_action_cluster_centers.append(pred_action_cluster_centers)
                # Set precise_optimization to True for the predicted sequence
                precise_optimization.append(True)  # Precise optimization is used
                batch_segments.append(segments_frame)        # Predicted segments
                batch_labels.append(labels_frame)            # Predicted labels

                labels_used_in_potential_actions = labels_frame
                segments_used_in_potential_actions = segments_frame

            elif use_multiple_possibilities == 'single_gt' or self.rebuild_model_cfg['debug_all_gt']:
                # === Ground Truth Sequence Processing ===
                # Using ground truth segments and ground truth labels (gt)
                gt_segments = single_video_list['segments']  # Ground truth segments
                gt_labels = single_video_list['labels']      # Ground truth labels

                # Convert ground truth segments and labels to frame-wise labels
                gt_pred = to_frame_wise(gt_segments, gt_labels, None, length)
                gt_labels_frame, gt_segments_frame = to_segments(gt_pred)

                gt_labels_frame = [0 if label == -1 else label for label in gt_labels_frame]

                if results_frame_type == 'gt':
                    results_frame[idx]['segments'] = torch.tensor(gt_segments_frame)
                    results_frame[idx]['labels'] = torch.tensor(gt_labels_frame)

                # Build error description features for the ground truth sequence
                gt_error_desc_feats = []
                gt_action_cluster_centers = []
                for action_id in gt_pred:
                    action_id = int(action_id)
                    action_id = 0 if action_id == -1 else action_id
                    error_desc = action_id_to_str[action_id]
                    if error_descriptions_to_feat is not None:
                        error_desc_feat = error_descriptions_to_feat[error_desc]
                        gt_error_desc_feats.append(error_desc_feat)
                    gt_action_cluster_centers.append(self.action_clusters.clusters_centers[action_id])
                gt_error_desc_feats = torch.cat(gt_error_desc_feats, dim=0).permute(1, 0)  if gt_error_desc_feats else None
                gt_action_cluster_centers = torch.stack(gt_action_cluster_centers, dim=0).permute(1, 0)
                batch_pred_error_desc_feats.append(gt_error_desc_feats)
                batch_pred_action_cluster_centers.append(gt_action_cluster_centers)
                # Set precise_optimization to True for the ground truth sequence
                precise_optimization.append(True)  # Precise optimization is used
                batch_segments.append(gt_segments_frame)     # Ground truth segments
                batch_labels.append(gt_labels_frame)         # Ground truth labels

                labels_used_in_potential_actions = gt_labels_frame
                segments_used_in_potential_actions = gt_segments_frame

            elif use_multiple_possibilities == 'single_intersection':
                pred_error_desc_feats = []
                pred_action_cluster_centers = []
                for action_id in pred:
                    error_desc = action_id_to_str[int(action_id)]
                    if error_descriptions_to_feat is not None:
                        error_desc_feat = error_descriptions_to_feat[error_desc]
                        pred_error_desc_feats.append(error_desc_feat)
                    pred_action_cluster_centers.append(self.action_clusters.clusters_centers[int(action_id)])
                pred_error_desc_feats = torch.cat(pred_error_desc_feats, dim=0).permute(1, 0)  if pred_error_desc_feats else None
                pred_action_cluster_centers = torch.stack(pred_action_cluster_centers, dim=0).permute(1, 0)
                batch_pred_error_desc_feats.append(pred_error_desc_feats)
                batch_pred_action_cluster_centers.append(pred_action_cluster_centers)
                precise_optimization.append(True)

                # Process each predicted segment against all GT segments to find intersections
                segments_selection = []
                labels_selection = []
                for idx, (pred_start, pred_end) in enumerate(segments_frame):
                    include_segment = False
                    for gt_start, gt_end in single_video_list['segments']:
                        if calculate_intersection_ratio(pred_start, pred_end, gt_start, gt_end) >= intersection_threshold:
                            include_segment = True
                            break  # If one valid intersection is found, no need to check further
                    if include_segment:
                        # Process this pred segment as it has a valid intersection
                        segments_selection.append(segments_frame[idx])        # Predicted segments with valid intersection
                        labels_selection.append(labels_frame[idx])            # Corresponding labels

                batch_segments.append(segments_selection)
                batch_labels.append(labels_selection)

                labels_used_in_potential_actions = labels_selection
                segments_used_in_potential_actions = segments_selection

            elif use_multiple_possibilities == 'single_intersection_gt_label':
                labels_used_in_potential_actions = labels_frame
                segments_used_in_potential_actions = segments_frame


                participated_segment_idx = []
                pred_error_desc_feats = []
                pred_action_cluster_centers = []
                for action_id in pred:
                    if action_id_to_str is not None:
                        error_desc = action_id_to_str[int(action_id)]
                    else:
                        error_desc = None
                    if error_descriptions_to_feat is not None:
                        error_desc_feat = error_descriptions_to_feat[error_desc]
                        pred_error_desc_feats.append(error_desc_feat)
                    temp_cluster_center = self.action_clusters.clusters_centers[int(action_id)]
                    if temp_cluster_center is not None:
                        pred_action_cluster_centers.append(temp_cluster_center)
                    else:
                        # If the current cluster center is None, use the mean of other non-None cluster centers
                        valid_centers = []
                        for i in range(len(self.action_clusters.clusters_centers)):
                            if self.action_clusters.clusters_centers[i] is not None:
                                valid_centers.append(self.action_clusters.clusters_centers[i])
                        if valid_centers:
                            mean_center = torch.stack(valid_centers).mean(dim=0)
                            pred_action_cluster_centers.append(mean_center)
                            logging.warning(f'Using mean of other cluster centers to replace None value for action_id: {action_id}')
                        else:
                            raise ValueError('All cluster centers are None, cannot compute mean')
                pred_error_desc_feats = torch.cat(pred_error_desc_feats, dim=0).permute(1, 0)  if pred_error_desc_feats else None
                pred_action_cluster_centers = torch.stack(pred_action_cluster_centers, dim=0).permute(1, 0)
                precise_optimization.append(True)

                # Convert GT segments and labels to frame-wise labels
                segments_labels_gt = to_frame_wise(single_video_list['segments'], single_video_list['labels'], None, length)
                if self.rebuild_model_cfg['dataset_name'] == 'HoloAssist':
                    segments_labels_error_gt = to_frame_wise(single_video_list['segments'], single_video_list['labels_error'], None, length)
                elif self.rebuild_model_cfg['dataset_name'] == 'CaptainCook4D':
                    segments_labels_error_gt = to_frame_wise(single_video_list['segments'], single_video_list['labels_error'], None, length)

                # Process each predicted segment against all GT segments to find intersections
                segments_selection = []
                labels_selection = []
                for idx, (pred_start, pred_end) in enumerate(segments_frame):
                    if pred_end > single_video_list['segments'][-1,1]:
                        if not labels_selection:
                            logging.info('labels_selection is empty')
                            logging.info(f'video_id: {single_video_list["video_id"]}, pred_end: {pred_end}, gt_end: {single_video_list["segments"][-1,1]}')
                            logging.info(f'segments_frame: {segments_frame}')
                            logging.info(f'labels_frame: {labels_frame}')
                            insert_empty_label_flag = True
                        break
                    include_segment = False
                    for gt_start, gt_end in single_video_list['segments']:
                        if calculate_intersection_ratio(pred_start, pred_end, gt_start, gt_end) >= intersection_threshold:
                            include_segment = True
                            break  # If one valid intersection is found, no need to check further
                    if include_segment:
                        # Find the majority GT label within the predicted segment
                        segment_labels_gt = segments_labels_gt[pred_start:pred_end + 1]
                        majority_label = most_common_element(segment_labels_gt.tolist())
                        # HoloAssist training samples have erroneous actions, filter them out
                        if self.rebuild_model_cfg['dataset_name'] == 'HoloAssist':
                            segment_labels_error_gt = segments_labels_error_gt[pred_start:pred_end + 1]
                            majority_label_error = most_common_element(segment_labels_error_gt.tolist())
                            if majority_label_error != 0 and self.rebuild_model_cfg['only_use_normal_action']:
                                # logging.info(f'[skip] majority_label_error: {majority_label_error} in {single_video_list["video_id"]}, start: {pred_start}, end: {pred_end}')
                                continue

                        # Process this pred segment as it has a valid intersection
                        segments_selection.append(segments_frame[idx])  # Predicted segments with valid intersection
                            
                        for i in range(pred_start, pred_end + 1):
                            if error_descriptions_to_feat is not None:
                                pred_error_desc_feats[:, i] = error_descriptions_to_feat[action_id_to_str[majority_label]]
                            
                            # Check if cluster center is None
                            cluster_center = self.action_clusters.clusters_centers[majority_label]
                            if cluster_center is None:
                                # If the current cluster center is None, use the mean of other non-None cluster centers
                                valid_centers = []
                                for j in range(len(self.action_clusters.clusters_centers)):
                                    if self.action_clusters.clusters_centers[j] is not None:
                                        valid_centers.append(self.action_clusters.clusters_centers[j])
                                if valid_centers:
                                    mean_center = torch.stack(valid_centers).mean(dim=0)
                                    pred_action_cluster_centers[:, i] = mean_center
                                    logging.warning(f'Using mean of other cluster centers to replace None value for majority_label: {majority_label}')
                                else:
                                    raise ValueError('All cluster centers are None, cannot compute mean')
                            else:
                                pred_action_cluster_centers[:, i] = cluster_center
                            
                        labels_selection.append(majority_label)  # Majority GT label for the segment

                        labels_used_in_potential_actions[idx] = majority_label
                        participated_segment_idx.append(idx)

                batch_pred_error_desc_feats.append(pred_error_desc_feats)
                batch_pred_action_cluster_centers.append(pred_action_cluster_centers)
                batch_segments.append(segments_selection)
                batch_labels.append(labels_selection)



            # === Handling Additional Sequences During Evaluation ===
            if max_added_seq > 0:
                potential_actions_seq = [[0] * max_added_seq] if not self.training else []
                if self.rebuild_model_cfg['dataset_name'] == 'HoloAssist':
                    task_name = f'{self.rebuild_model_cfg["task_name"]}_{self.rebuild_model_cfg["segment_type"]}'
                else:
                    task_name = single_video_list['video_id'].split('_')[0]
                action_idx_list = range(1, len(labels_used_in_potential_actions)) if not self.training else participated_segment_idx
                for action_index in action_idx_list:
                    past_action_seq = labels_used_in_potential_actions[:action_index]
                    if self.potential_action_generation_filtration_rate > 0:
                        past_action_segments = segments_used_in_potential_actions[:action_index]
                        past_action_seq, _ = filter_actions(past_action_seq, past_action_segments, self.potential_action_generation_filtration_rate)
                    
                    if self.rebuild_model_cfg['debug_all_gt']:
                        segments_labels_gt = to_frame_wise(single_video_list['segments'], single_video_list['labels'], None, length)
                        segments_labels_gt = [0 if label == -1 else label for label in segments_labels_gt]
                        start = segments_used_in_potential_actions[action_index][0]
                        end = segments_used_in_potential_actions[action_index][1]
                        segment_labels_gt = segments_labels_gt[start:end+1]
                        gt_label = most_common_element(segment_labels_gt)
                        seq = [int(gt_label)]
                    else:
                        pred_action = labels_used_in_potential_actions[action_index]
                        for i in range(action_index-1, -1, -1):
                            if labels_used_in_potential_actions[i] != 0:
                                fill_action = labels_used_in_potential_actions[i]
                                if fill_action == pred_action:
                                    fill_action = 0
                                break
                        else:
                            fill_action = 0
                        
                        if use_random_uncertainty:
                            # Randomly generate possible actions
                            all_actions = list(range(1, self.num_classes))
                            if pred_action in all_actions:
                                all_actions.remove(pred_action)
                            potential_actions = random.sample(all_actions, min(max_added_seq + 1, len(all_actions)))
                        else:
                            # Get possible actions from the task graph
                            if self.rebuild_model_cfg['bfs_find_next_actions']:
                                potential_actions = self.bfs_find_next_actions(task_name, past_action_seq)
                            else:
                                potential_actions = self.get_all_valid_next_actions(task_name, past_action_seq)
                            
                        # Separate action class and background class
                        if self.rebuild_model_cfg['separate_action_and_bg']:
                            if 0 in potential_actions:
                                potential_actions.remove(0)
                        else:
                            # Do not separate action class and background class, check if background class is added
                            if 0 not in potential_actions and fill_action != 0:
                                potential_actions += [0]
                        potential_actions += [fill_action]
                        # Remove the predicted class
                        if pred_action in potential_actions:
                            potential_actions.remove(pred_action)

                       
                        all_actions = list(range(1, self.num_classes))
                        if pred_action in all_actions:
                            all_actions.remove(pred_action)
                        all_actions = [action for action in all_actions if action not in potential_actions]
                         # Sample max_added_seq actions from potential_actions and add to potential_actions_seq
                        if potential_actions:
                            if fill_action in potential_actions:
                                # logging.info(f'[analysis] len of potential_actions: {len(potential_actions)}')
                                potential_actions.remove(fill_action)
                                seq = random.sample(potential_actions, min(max_added_seq - 1, len(potential_actions)))
                                seq.append(fill_action)
                            else:
                                seq = random.sample(potential_actions, min(max_added_seq, len(potential_actions)))

                            if self.training:
                                # During training phase, randomly sample from all actions to fill the remaining slots
                                remaining = max_added_seq - len(seq)
                                if remaining > 0:
                                    seq += random.sample(all_actions, remaining)
                            else:
                                # During testing phase, fill with fill_action
                                seq += [fill_action] * (max_added_seq - len(seq))
                        else:
                            if self.training:
                                seq = random.sample(all_actions, max_added_seq)
                            else:
                                seq = [fill_action] * max_added_seq

                        # logging.info(f'{pred_action} {potential_actions} {seq}')
                        # logging.info(f'{labels_used_in_potential_actions[:action_index]}\n')
                    
                    potential_actions_seq.append(seq)
                

                if not self.training:
                    added_segments = segments_used_in_potential_actions
                    pred_error_desc_feats_list = [[] for _ in range(max_added_seq)]
                    pred_action_cluster_centers_list = [[] for _ in range(max_added_seq)]
                    for actions, segment in zip(potential_actions_seq, segments_used_in_potential_actions):
                        for temporal_idx in range(segment[0], segment[1] + 1):
                            for insert_sample_idx, action in enumerate(actions):
                                error_desc = action_id_to_str[int(action)]
                                if error_descriptions_to_feat is not None:
                                    error_desc_feat = error_descriptions_to_feat[error_desc]
                                    pred_error_desc_feats_list[insert_sample_idx].append(error_desc_feat)
                                pred_action_cluster_centers_list[insert_sample_idx].append(self.action_clusters.clusters_centers[action])

                    for i in range(max_added_seq):
                        pred_error_desc_feats_list[i] = torch.cat(pred_error_desc_feats_list[i], dim=0).permute(1, 0) if pred_error_desc_feats_list[i] else None
                        pred_action_cluster_centers_list[i] = torch.stack(pred_action_cluster_centers_list[i], dim=0).permute(1, 0)
                else:
                    added_segments = []
                    pred_error_desc_feats_list = [pred_error_desc_feats for _ in range(max_added_seq)]
                    pred_action_cluster_centers_list = [pred_action_cluster_centers for _ in range(max_added_seq)]
                    for actions, action_idx in zip(potential_actions_seq, action_idx_list):
                        segment = segments_used_in_potential_actions[action_idx]
                        added_segments.append(segment)
                        for insert_sample_idx, action in enumerate(actions):
                            error_desc = action_id_to_str[int(action)]
                            error_desc_feat = error_descriptions_to_feat[error_desc] if error_descriptions_to_feat is not None else None
                            for frame_idx in range(segment[0], segment[1] + 1):
                                if error_desc_feat is not None:
                                    pred_error_desc_feats_list[insert_sample_idx][:, frame_idx] = error_desc_feat[0]
                                pred_action_cluster_centers_list[insert_sample_idx][:, frame_idx] = self.action_clusters.clusters_centers[action]



                transposed_potential_actions_seq = list(map(list, zip(*potential_actions_seq)))


                batch_pred_error_desc_feats.extend(pred_error_desc_feats_list)
                batch_pred_action_cluster_centers.extend(pred_action_cluster_centers_list)
                batch_segments.extend([added_segments] * max_added_seq)
                batch_labels.extend(transposed_potential_actions_seq)
                if insert_empty_label_flag:
                    logging.info(f'batch_labels is empty, insert {max_added_seq} empty labels')
                    for i in range(max_added_seq):
                        batch_labels.append([])
                if self.training:
                    precise_optimization.extend([False] * max_added_seq)
                else:
                    precise_optimization.extend([True] * max_added_seq)



        # Stack all error description features into a batch
        batch_pred_error_desc_feats = torch.stack(batch_pred_error_desc_feats, dim=0).to(self.device) if all(x is not None for x in batch_pred_error_desc_feats) else None
        batch_pred_action_cluster_centers = torch.stack(batch_pred_action_cluster_centers, dim=0).to(self.device)

        # Adjust fpn_feats and fpn_masks to match the number of sequences per sample
        fpn_feats_potential_sequences = [t.clone() for t in fpn_feats]
        fpn_masks_potential_sequences = [t.clone() for t in fpn_masks]


        # === Expand FPN Features and Masks to Match the Number of Sequences ===
        if max_added_seq > 0:
            # Calculate the total number of sequences per original sample
            total_sequences_multiplier = 1 + max_added_seq

            # Expand each FPN feature map
            for level in range(len(fpn_feats_potential_sequences)):
                # Original shape: [B, C, T_i]
                # New shape: [B * (1 + max_added_seq), C, T_i]
                fpn_feats_potential_sequences[level] = fpn_feats_potential_sequences[level].repeat_interleave(total_sequences_multiplier, dim=0)

            # Expand each FPN mask
            for level in range(len(fpn_masks_potential_sequences)):
                # Original shape: [B, T_i]
                # New shape: [B * (1 + max_added_seq), T_i]
                fpn_masks_potential_sequences[level] = fpn_masks_potential_sequences[level].repeat_interleave(total_sequences_multiplier, dim=0)

        return (fpn_feats_potential_sequences, fpn_masks_potential_sequences, batch_pred_error_desc_feats, batch_pred_action_cluster_centers,
                batch_segments, batch_labels, precise_optimization, results_frame)





    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        labels_error = [x['labels_error'] for x in video_list]
        # vid_ft_stride = [x['feat_stride'] for x in video_list]
        # vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        # for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
        #     zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        # ):
        for idx, (vidx, fps, vlen) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['labels_error'] = labels_error[idx]
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []
        org_scores_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
            # 1004
            org_scores_all.append(pred_prob)

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])
        
        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all,
                   'scores_all': org_scores_all} # 1004

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            scores_all = results_per_vid['scores_all']
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )

                # 3: convert from feature grids to seconds
                if segs.shape[0] > 0:
                    segs = segs / fps
                    # truncate all boundaries within [0, duration]
                    segs[segs<=0.0] *= 0.0
                    segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen

            else: # just return the results of original length
                num_frames = int(vlen * fps)
                segs = segs[:num_frames,:]
                scores = scores[:num_frames]
                labels = labels[:num_frames]
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels,
                 'scores_all': scores_all,
                 'labels_error': results_per_vid['labels_error'],
                 'fps'      : fps,}
            )

        return processed_results