import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import CausalLocalMultiHeadSelfConvAttention, CausalLocalMultiHeadCrossConvAttention

class MultimodalAttentionModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_heads=2, win_len=32, num_layers=1, future_step=1, dilation=2, dilated_conv_ks=3, dilated_conv_layers=3, **kwargs):
        super(MultimodalAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.num_classes = num_classes

        # Encoders for visual and text features before attention and interaction
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.ReLU()
        )

        # Causal Dilated Conv
        self.dilated_conv_ks = dilated_conv_ks
        self.dilated_conv_layers = dilated_conv_layers
        self.dilation = dilation
        # Multi-layer Causal Dilated Conv
        self.causal_dilated_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.dilated_conv_ks, padding=self.dilation**i * (self.dilated_conv_ks - 1), dilation=self.dilation**i)
            for i in range(self.dilated_conv_layers)
        ])

        # Attention mechanism for visual features
        self.attentions = nn.ModuleList([
            CausalLocalMultiHeadSelfConvAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                win_len=win_len, 
                n_qx_stride=1, 
                n_kv_stride=1, 
                attn_pdrop=0.0, 
                proj_pdrop=0.0
            ) for _ in range(num_layers)
        ])
        
        # Intermediate fully connected layers for attention outputs
        self.intermediate_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        # Fully connected layer for combining attention output and text features
        self.fc1 = nn.Linear(hidden_dim + text_dim, hidden_dim)
        
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, visual_features, text_features, mask):
        batch_size, visual_dim, seq_length = visual_features.size()
        _, text_dim, _ = text_features.size()
        
        # Transpose to shape (B, D_V, T) and (B, D_T, T)
        visual_features = visual_features.transpose(1, 2)
        text_features = text_features.transpose(1, 2)
        mask = mask.unsqueeze(2)  # Shape (B, T, 1)

        # Apply mask to visual and text features
        visual_features = visual_features * mask
        text_features = text_features * mask
        
        # Encode visual and text features
        visual_features = self.visual_encoder(visual_features)
        text_features = self.text_encoder(text_features)
        
        # Causal Dilate Conv
        visual_features = visual_features.transpose(1, 2)
        for i, conv in enumerate(self.causal_dilated_convs):
            visual_features = conv(visual_features)
            visual_features = visual_features[:, :, :-((self.dilated_conv_ks - 1) * self.dilation ** i)]

        # Pass visual features through each attention layer with intermediate layers
        for attn, fc in zip(self.attentions, self.intermediate_fcs):
            visual_features, _ = attn(visual_features)
            visual_features = fc(visual_features.transpose(1, 2)).transpose(1, 2)
        visual_features = visual_features.transpose(1, 2)
        
        # Extract relevant output from attention and corresponding future text features
        attention_last_output = visual_features[:, :-self.future_step, :]
        text_future = text_features[:, self.future_step:, :]

        # Concatenate attention output and future text features
        combined_features = torch.cat((attention_last_output, text_future), dim=2)

        # Pass through fully connected layers
        out = self.relu(self.fc1(combined_features))
        
        output_features = self.fc2(out)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask.transpose(1, 2)
        
        if self.num_classes > 0:
            output_logits = self.classifier(out)
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features



class MultimodalCrossAttentionModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_heads=2, 
                 win_len=32, num_layers=1, future_step=1, dilation=2, dilated_conv_ks=3, dilated_conv_layers=3, 
                 concat_query_to_attnout=False, drop_rate=0.0, prompt_len=None, noise_std=None, conv_non_linear=False, **kwargs):
        super(MultimodalCrossAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.num_classes = num_classes
        self.prompt_len = prompt_len
        self.add_noise = True if noise_std is not None else False
        self.noise_std = noise_std
        self.conv_non_linear = conv_non_linear

        if self.prompt_len:
            self.prompt = nn.Parameter(torch.randn(visual_dim, self.prompt_len))

        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )


        # self.dropout_visual = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()
        # self.dropout_text = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()


        # Causal Dilated Conv
        self.dilated_conv_ks = dilated_conv_ks
        self.dilated_conv_layers = dilated_conv_layers
        self.dilation = dilation
        # Multi-layer Causal Dilated Conv
        self.causal_dilated_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.dilated_conv_ks, padding=self.dilation**i * (self.dilated_conv_ks - 1), dilation=self.dilation**i)
            for i in range(self.dilated_conv_layers)
        ])

        # Attention mechanism for visual features
        self.attentions = nn.ModuleList([
            CausalLocalMultiHeadCrossConvAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                win_len=win_len, 
                n_qx_stride=1, 
                n_kv_stride=1, 
                attn_pdrop=drop_rate, 
                proj_pdrop=drop_rate
            ) for _ in range(num_layers)
        ])
        
        # Intermediate fully connected layers for attention outputs
        self.intermediate_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        self.concat_query_to_attnout = concat_query_to_attnout
        if concat_query_to_attnout:
            self.mlp_query = nn.Identity()
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)

        # self.dropout_out = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.mlp_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, visual_features, text_features, mask):
        batch_size, visual_dim, seq_length = visual_features.size()
        _, text_dim, _ = text_features.size()


        noise = None
        if self.add_noise:
            noise = torch.randn_like(text_features) * self.noise_std
            text_features = text_features + noise
        
        # Transpose to shape (B, D_V, T) and (B, D_T, T)
        visual_features = visual_features.transpose(1, 2)
        text_features = text_features.transpose(1, 2)
        mask = mask.unsqueeze(2)  # Shape (B, T, 1)
        
        # Apply mask to visual and text features
        visual_features = visual_features * mask
        text_features = text_features * mask
        
        if self.concat_query_to_attnout:
            text_features_original = text_features.clone()[:,self.future_step:,:]
            # visual_features = torch.nn.functional.normalize(visual_features, p=2, dim=-1) 
            # text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        
        # Add prompt
        if self.prompt_len:
            prompt_batch = self.prompt.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2)  # (B, self.prompt_len, vision_feat_d)
            visual_features = torch.cat((prompt_batch, visual_features), dim=1)  # Concatenate along time dimension
            text_features = F.pad(text_features, (0, 0, self.prompt_len, 0))
    
        # Encode visual and text features
        visual_features = self.visual_encoder(visual_features)
        text_features = self.text_encoder(text_features)
        
        # Causal Dilate Conv
        visual_features = visual_features.transpose(1, 2)
        for i, conv in enumerate(self.causal_dilated_convs):
            visual_features = conv(visual_features)
            if self.conv_non_linear:
                visual_features = F.relu(visual_features)
            visual_features = visual_features[:, :, :-((self.dilated_conv_ks - 1) * self.dilation ** i)]
        

        # Extract relevant output from attention and corresponding future text features
        visual_features = visual_features[:, :, :-self.future_step]
        text_features = text_features[:, self.future_step:, :].transpose(1, 2)

        # Pass visual features through each attention layer with intermediate layers
        for attn, fc in zip(self.attentions, self.intermediate_fcs):
            visual_features, _ = attn(text_features, visual_features, visual_features)
            visual_features = fc(visual_features.transpose(1, 2)).transpose(1, 2)

        if self.prompt_len:
            visual_features = visual_features[:,:,self.prompt_len:]
            text_features = text_features[:,:,self.prompt_len:]

        # Pass through fully connected layers
        if self.concat_query_to_attnout:
            text_features = self.mlp_query(text_features_original)
            visual_features = torch.nn.functional.normalize(visual_features.transpose(1, 2), p=2, dim=-1) 
            output_features = torch.cat((visual_features, text_features), dim=2)
        else:
            output_features = visual_features.transpose(1, 2)
        output_features = self.fc1(output_features)
        # output_features = self.dropout_out(output_features)
        output_features = self.mlp_out(output_features)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        if self.add_noise:
            output_features = output_features - noise
        output_features = output_features * mask.transpose(1, 2)


        if self.num_classes > 0:
            output_logits = self.classifier(visual_features)
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features



class MultimodalSelfCrossAttentionModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_heads=2, 
                 win_len=32, num_layers=1, future_step=1, dilation=2, dilated_conv_ks=3, dilated_conv_layers=3, 
                 concat_query_to_attnout=False, drop_rate=0.0, prompt_len=None, noise_std=None, conv_non_linear=False, **kwargs):
        super(MultimodalSelfCrossAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.num_classes = num_classes
        self.prompt_len = prompt_len
        self.add_noise = True if noise_std is not None else False
        self.noise_std = noise_std
        self.conv_non_linear = conv_non_linear

        if self.prompt_len:
            self.prompt = nn.Parameter(torch.randn(visual_dim, self.prompt_len))

        # Encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Causal dilated convolution
        self.dilated_conv_ks = dilated_conv_ks
        self.dilated_conv_layers = dilated_conv_layers
        self.dilation = dilation
        # Multi-layer Causal Dilated Conv
        self.causal_dilated_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.dilated_conv_ks, padding=self.dilation**i * (self.dilated_conv_ks - 1), dilation=self.dilation**i)
            for i in range(self.dilated_conv_layers)
        ])

        # Self Attention layer
        self.self_attentions = nn.ModuleList([
            CausalLocalMultiHeadSelfConvAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                win_len=win_len,
                n_qx_stride=1,
                n_kv_stride=1,
                attn_pdrop=drop_rate,
                proj_pdrop=drop_rate
            ) for _ in range(num_layers)
        ])
        
        # Self Attention layer after
        self.self_intermediate_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        # Cross Attention layer
        self.cross_attentions = nn.ModuleList([
            CausalLocalMultiHeadCrossConvAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                win_len=win_len,
                n_qx_stride=1,
                n_kv_stride=1,
                attn_pdrop=drop_rate,
                proj_pdrop=drop_rate
            ) for _ in range(num_layers)
        ])
        
        # Cross Attention layer after
        self.cross_intermediate_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        # Output layer
        self.concat_query_to_attnout = concat_query_to_attnout
        if concat_query_to_attnout:
            self.mlp_query = nn.Identity()
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)

        self.mlp_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, visual_features, text_features, mask):
        batch_size, visual_dim, seq_length = visual_features.size()
        _, text_dim, _ = text_features.size()

        noise = None
        if self.add_noise:
            noise = torch.randn_like(text_features) * self.noise_std
            text_features = text_features + noise
        
        # Transpose to shape (B, D_V, T) and (B, D_T, T)
        visual_features = visual_features.transpose(1, 2)
        text_features = text_features.transpose(1, 2)
        mask = mask.unsqueeze(2)
        
        visual_features = visual_features * mask
        text_features = text_features * mask
        
        if self.concat_query_to_attnout:
            text_features_original = text_features.clone()[:,self.future_step:,:]
        
        # Add prompt
        if self.prompt_len:
            prompt_batch = self.prompt.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2)
            visual_features = torch.cat((prompt_batch, visual_features), dim=1)
            text_features = F.pad(text_features, (0, 0, self.prompt_len, 0))
    
        # Encode
        visual_features = self.visual_encoder(visual_features)
        text_features = self.text_encoder(text_features)
        
        # Causal Dilate Conv
        visual_features = visual_features.transpose(1, 2)
        for i, conv in enumerate(self.causal_dilated_convs):
            visual_features = conv(visual_features)
            if self.conv_non_linear:
                visual_features = F.relu(visual_features)
            visual_features = visual_features[:, :, :-((self.dilated_conv_ks - 1) * self.dilation ** i)]
        
        # Extract relevant output from attention and corresponding future text features
        visual_features = visual_features[:, :, :-self.future_step]
        text_features = text_features[:, self.future_step:, :].transpose(1, 2)

        # First perform Self Attention
        for self_attn, self_fc in zip(self.self_attentions, self.self_intermediate_fcs):
            visual_features, _ = self_attn(visual_features)
            visual_features = self_fc(visual_features.transpose(1, 2)).transpose(1, 2)

        # Then perform Cross Attention
        for cross_attn, cross_fc in zip(self.cross_attentions, self.cross_intermediate_fcs):
            visual_features, _ = cross_attn(text_features, visual_features, visual_features)
            visual_features = cross_fc(visual_features.transpose(1, 2)).transpose(1, 2)

        if self.prompt_len:
            visual_features = visual_features[:,:,self.prompt_len:]
            text_features = text_features[:,:,self.prompt_len:]

        # Output processing
        if self.concat_query_to_attnout:
            text_features = self.mlp_query(text_features_original)
            visual_features = torch.nn.functional.normalize(visual_features.transpose(1, 2), p=2, dim=-1)
            output_features = torch.cat((visual_features, text_features), dim=2)
        else:
            output_features = visual_features.transpose(1, 2)
            
        output_features = self.fc1(output_features)
        output_features = self.mlp_out(output_features)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        if self.add_noise:
            output_features = output_features - noise
        output_features = output_features * mask.transpose(1, 2)

        if self.num_classes > 0:
            output_logits = self.classifier(visual_features.transpose(1, 2))
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features



class MultimodalTwoStreamsCrossAttentionModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_heads=2, win_len=32, win_len_large=None, num_layers=1, future_step=1, dilation=2, dilated_conv_ks=3, dilated_conv_layers=3, **kwargs):
        super(MultimodalTwoStreamsCrossAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.num_classes = num_classes

        # Encoders for visual and text features before attention and interaction
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Causal Dilated Conv
        self.dilated_conv_ks = dilated_conv_ks
        self.dilated_conv_layers = dilated_conv_layers
        self.dilation = dilation
        # Multi-layer Causal Dilated Conv
        self.causal_dilated_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.dilated_conv_ks, padding=self.dilation**i * (self.dilated_conv_ks - 1), dilation=self.dilation**i)
            for i in range(self.dilated_conv_layers)
        ])


        if win_len_large is None:
            self.win_len_large = win_len * 4
        else:
            self.win_len_large = win_len_large
        
        # Attention mechanism for visual features
        self.attentions_small = nn.ModuleList([
            CausalLocalMultiHeadCrossConvAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                win_len=win_len, 
                n_qx_stride=1, 
                n_kv_stride=1, 
                attn_pdrop=0.0, 
                proj_pdrop=0.0
            ) for _ in range(num_layers)
        ])
        self.intermediate_fcs_small = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        self.attentions_large = nn.ModuleList([
            CausalLocalMultiHeadCrossConvAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                win_len=self.win_len_large, 
                n_qx_stride=1, 
                n_kv_stride=1, 
                attn_pdrop=0.0, 
                proj_pdrop=0.0
            ) for _ in range(num_layers)
        ])
        self.intermediate_fcs_large = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        # Fully connected layer for combining attention output and text features
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, visual_features, text_features, mask):
        batch_size, visual_dim, seq_length = visual_features.size()
        _, text_dim, _ = text_features.size()
        
        # Transpose to shape (B, D_V, T) and (B, D_T, T)
        visual_features = visual_features.transpose(1, 2)
        text_features = text_features.transpose(1, 2)
        mask = mask.unsqueeze(2)  # Shape (B, T, 1)

        # Apply mask to visual and text features
        visual_features = visual_features * mask
        text_features = text_features * mask
        
        # Encode visual and text features
        visual_features = self.visual_encoder(visual_features)
        text_features = self.text_encoder(text_features)
        
        # Causal Dilate Conv
        visual_features = visual_features.transpose(1, 2)
        for i, conv in enumerate(self.causal_dilated_convs):
            visual_features = conv(visual_features)
            visual_features = visual_features[:, :, :-((self.dilated_conv_ks - 1) * self.dilation ** i)]
        
        # Extract relevant output from attention and corresponding future text features
        visual_features = visual_features[:, :, :-self.future_step]
        text_features = text_features[:, self.future_step:, :].transpose(1, 2)


        # Pass visual features through each attention layer with intermediate layers
        visual_features_small = visual_features.clone()
        text_features_small = text_features.clone()
        for attn, fc in zip(self.attentions_small, self.intermediate_fcs_small):
            visual_features_small, _ = attn(text_features_small, visual_features_small, visual_features_small)
            visual_features_small = fc(visual_features_small.transpose(1, 2)).transpose(1, 2)

        visual_features_large = visual_features.clone()
        text_features_large = text_features.clone()
        for attn, fc in zip(self.attentions_large, self.intermediate_fcs_large):
            visual_features_large, _ = attn(text_features_large, visual_features_large, visual_features_large)
            visual_features_large = fc(visual_features_large.transpose(1, 2)).transpose(1, 2)

        visual_features = torch.cat((visual_features_small.transpose(1, 2), visual_features_large.transpose(1, 2)), dim=2)

        # Pass through fully connected layers
        output_features = self.relu(self.fc1(visual_features))        
        output_features = self.fc2(output_features)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask.transpose(1, 2)
        
        if self.num_classes > 0:
            output_logits = self.classifier(visual_features)
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features





class VisualAttentionModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_heads=2, 
                 win_len=32, num_layers=1, future_step=1, dilation=2, dilated_conv_ks=3, dilated_conv_layers=3,
                 drop_rate=0.0, prompt_len=None, noise_std=None, conv_non_linear=False, **kwargs):
        super(VisualAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.num_classes = num_classes
        self.prompt_len = prompt_len
        self.add_noise = True if noise_std is not None else False
        self.noise_std = noise_std
        self.conv_non_linear = conv_non_linear

        if self.prompt_len:
            self.prompt = nn.Parameter(torch.randn(visual_dim, self.prompt_len))

        # Visual feature encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Causal dilated convolution
        self.dilated_conv_ks = dilated_conv_ks
        self.dilated_conv_layers = dilated_conv_layers
        self.dilation = dilation
        # Multi-layer Causal Dilated Conv
        self.causal_dilated_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.dilated_conv_ks,
                     padding=self.dilation**i * (self.dilated_conv_ks - 1),
                     dilation=self.dilation**i)
            for i in range(self.dilated_conv_layers)
        ])

        # Attention mechanism for visual features
        self.attentions = nn.ModuleList([
            CausalLocalMultiHeadSelfConvAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                win_len=win_len, 
                n_qx_stride=1, 
                n_kv_stride=1, 
                attn_pdrop=drop_rate, 
                proj_pdrop=drop_rate
            ) for _ in range(num_layers)
        ])
        
        # Intermediate fully connected layers for attention outputs
        self.intermediate_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        # Fully connected layers for producing the final output
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
            
        self.mlp_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, visual_features, text_features, mask):
        # Note: text_features is ignored in this version
        
        batch_size, visual_dim, seq_length = visual_features.size()
        
        # Transpose to shape (B, T, D_V)
        visual_features = visual_features.transpose(1, 2)
        mask = mask.unsqueeze(2)  # Shape (B, T, 1)

        # Apply mask to visual features
        visual_features = visual_features * mask
        
        # Add prompt if needed
        if self.prompt_len:
            prompt_batch = self.prompt.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2)
            visual_features = torch.cat((prompt_batch, visual_features), dim=1)
            mask = F.pad(mask, (0, 0, self.prompt_len, 0))
        
        # Encode visual features
        visual_features = self.visual_encoder(visual_features)
        
        # Causal Dilated Conv
        visual_features = visual_features.transpose(1, 2)  # Shape (B, D_V, T)
        for i, conv in enumerate(self.causal_dilated_convs):
            visual_features = conv(visual_features)
            if self.conv_non_linear:
                visual_features = F.relu(visual_features)
            visual_features = visual_features[:, :, :-((self.dilated_conv_ks - 1) * self.dilation ** i)]
        
        # 提取相关输出
        visual_features = visual_features[:, :, :-self.future_step]
        
        # 通过注意力层
        for attn, fc in zip(self.attentions, self.intermediate_fcs):
            visual_features, _ = attn(visual_features)
            visual_features = fc(visual_features.transpose(1, 2)).transpose(1, 2)

        if self.prompt_len:
            visual_features = visual_features[:,:,self.prompt_len:]
            
        # 通过全连接层
        output_features = visual_features.transpose(1, 2)
        output_features = self.fc1(output_features)
        output_features = self.mlp_out(output_features)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask.transpose(1, 2)
        
        if self.num_classes > 0:
            output_logits = self.classifier(visual_features.transpose(1, 2))
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features



class MultimodalLSTMModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_layers=1, future_step=1):
        super(MultimodalLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.num_classes = num_classes
        
        # Encoders for visual and text features before LSTM and interaction
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU()
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.ReLU()
        )
        
        # LSTM for time series visual features
        self.lstm = nn.LSTM(visual_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for combining LSTM output and text features
        self.fc1 = nn.Linear(hidden_dim + text_dim, hidden_dim)
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, visual_features, text_features, mask):
        batch_size, visual_dim, seq_length = visual_features.size()
        _, text_dim, _ = text_features.size()
        
        # Transpose to shape (B, T, D_V) and (B, T, D_T)
        visual_features = visual_features.transpose(1, 2)
        text_features = text_features.transpose(1, 2)
        mask = mask.unsqueeze(2) # Shape (B, T, 1)

        # Apply mask to visual and text features
        visual_features = visual_features * mask
        text_features = text_features * mask
        
        # Encode visual and text features
        visual_features = self.visual_encoder(visual_features)
        text_features = self.text_encoder(text_features)

        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(visual_features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(visual_features.device)
        
        # Pass visual features through LSTM
        lstm_out, _ = self.lstm(visual_features, (h0, c0))
        
        # Take the output of the LSTM up to the last required time step
        lstm_last_output = lstm_out[:, :-self.future_step, :] 
        text_future = text_features[:, self.future_step:, :]

        # Concatenate LSTM output and future text features
        combined_features = torch.cat((lstm_last_output, text_future), dim=2)

        # Pass through fully connected layers
        out = self.relu(self.fc1(combined_features))
        
        output_features = self.fc2(out)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask.transpose(1, 2)
        
        if self.num_classes > 0:
            output_logits = self.classifier(out)
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features

class VisualLSTMModel(nn.Module):
    def __init__(self, visual_dim, hidden_dim, output_dim, num_classes=-1, num_layers=1, future_step=1, **kwargs):
        super(VisualLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        
        # LSTM for time series visual features
        self.lstm = nn.LSTM(visual_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, visual_features, text_features, mask):
        batch_size, visual_dim, seq_length = visual_features.size()
        
        # Transpose to shape (B, T, D_V)
        visual_features = visual_features.transpose(1, 2)
        mask = mask.unsqueeze(2) # Shape (B, T, 1)

        # Apply mask to visual features
        visual_features = visual_features * mask
        
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(visual_features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(visual_features.device)
        
        # Pass visual features through LSTM
        lstm_out, _ = self.lstm(visual_features, (h0, c0))
        
        # Take the output of the LSTM up to the last required time step
        lstm_last_output = lstm_out[:, :-self.future_step, :]

        output_features = self.fc(lstm_last_output)
        output_features = output_features.transpose(1, 2)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask.transpose(1, 2)
        
        if self.num_classes > 0:
            output_logits = self.classifier(lstm_last_output)
            output_probs = output_logits
            output_probs = output_probs.transpose(1, 2)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask.transpose(1, 2)
            
            return output_probs, output_features
        else:
            return output_features


class MultimodalMLPModel(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim, num_classes=-1, num_layers=1, future_step=1, history_length=32):
        super(MultimodalMLPModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.history_length = history_length
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Calculate input dimension for historical visual features
        total_visual_dim = visual_dim * history_length
        
        # MLP model
        self.fc1 = nn.Linear(total_visual_dim + text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Can add more layers to increase model complexity
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, visual_features, text_features, mask):
        """
        Args:
            visual_features: Tensor of shape (B, D_V, T)
            text_features: Tensor of shape (B, D_T, T)
            
        Returns:
            output_features: Tensor of shape (B, output_dim, T) or (B, num_classes, T)
        """
        batch_size, visual_dim, seq_length = visual_features.size()
        _, text_dim, _ = text_features.size()
        
        # Calculate the range of required time steps
        start_idx = self.history_length + self.future_step - 1
        
        mask = mask.unsqueeze(1)  # Shape (B, 1, T)
        visual_features = visual_features * mask
        text_features = text_features * mask

        # padding visual_features
        visual_features = F.pad(visual_features, (self.history_length - 1, 0, 0, 0))

        # Reshape visual features to (B, T, D_V * history_length)
        visual_past = []
        for t in range(start_idx, visual_features.shape[-1]):
            visual_past.append(visual_features[:, :, t-start_idx:t-self.future_step+1].reshape(batch_size, -1))
        visual_past = torch.stack(visual_past, dim=1)
        

        # Extract text features for corresponding time steps (B, T - start_idx, D_T)
        text_current = text_features[:, :, self.future_step:].permute(0, 2, 1)
        
        # Combine visual and text features (B, T - start_idx, D_V * history_length + D_T)
        combined_features = torch.cat((visual_past, text_current), dim=2)

        # Process through MLP
        out = self.relu(self.fc1(combined_features))
        out = self.relu(self.fc2(out))
        
        output_features = self.fc3(out)
        output_features = output_features.permute(0, 2, 1)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask
        
        if self.num_classes > 0:
            output_logits = self.classifier(out)
            output_probs = output_logits
            output_probs = output_probs.permute(0, 2, 1)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask
            
            return output_probs, output_features
        else:
            return output_features

class VisualMLPModel(nn.Module):
    def __init__(self, visual_dim, hidden_dim, output_dim, num_classes=-1, num_layers=1, future_step=1, history_length=32, **kwargs):
        super(VisualMLPModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_step = future_step
        self.history_length = history_length
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Calculate input dimension for historical visual features
        total_visual_dim = visual_dim * history_length
        
        # MLP model
        self.fc1 = nn.Linear(total_visual_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Can add more layers to increase model complexity
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, visual_features, text_features, mask):
        """
        Args:
            visual_features: Tensor of shape (B, D_V, T)
            text_features: Tensor of shape (B, D_T, T) (unused in this model)
            
        Returns:
            output_features: Tensor of shape (B, output_dim, T) or (B, num_classes, T)
        """
        batch_size, visual_dim, seq_length = visual_features.size()
        
        # Calculate the range of required time steps
        start_idx = self.history_length + self.future_step - 1
        
        mask = mask.unsqueeze(1)  # Shape (B, 1, T)
        visual_features = visual_features * mask

        # padding visual_features
        visual_features = F.pad(visual_features, (self.history_length - 1, 0, 0, 0))

        # Reshape visual features to (B, T, D_V * history_length)
        visual_past = []
        for t in range(start_idx, visual_features.shape[-1]):
            visual_past.append(visual_features[:, :, t-start_idx:t-self.future_step+1].reshape(batch_size, -1))
        visual_past = torch.stack(visual_past, dim=1)

        # Process through MLP
        out = self.relu(self.fc1(visual_past))
        out = self.relu(self.fc2(out))
        
        output_features = self.fc3(out)
        output_features = output_features.permute(0, 2, 1)
        output_features = F.pad(output_features, (self.future_step, 0, 0, 0))
        output_features = output_features * mask
        
        if self.num_classes > 0:
            output_logits = self.classifier(out)
            output_probs = output_logits
            output_probs = output_probs.permute(0, 2, 1)
            output_probs = F.pad(output_probs, (self.future_step, 0, 0, 0))
            output_probs = output_probs * mask
            
            return output_probs, output_features
        else:
            return output_features



class SimpleMLPModel(nn.Module):
    def __init__(self, visual_dim, hidden_dim, output_dim, num_classes=-1, num_layers=2, context_length=16, **kwargs):
        super(SimpleMLPModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_length = context_length
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Calculate input dimension for context visual features
        total_visual_dim = visual_dim * (2 * context_length + 1)
        
        # MLP model
        layers = [nn.Linear(total_visual_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.regressor = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, visual_features, text_features, mask):
        """
        Args:
            visual_features: Tensor of shape (B, D_V, T)
            text_features: Tensor of shape (B, D_T, T) (unused in this model)
            
        Returns:
            output_features: Tensor of shape (B, output_dim, T) or (B, num_classes, T)
        """
        batch_size, visual_dim, seq_length = visual_features.size()
        
        mask = mask.unsqueeze(1)  # Shape (B, 1, T)
        visual_features = visual_features * mask

        # padding visual_features
        visual_features = F.pad(visual_features, (self.context_length, self.context_length, 0, 0))

        # Extract context visual features (B, T, D_V * (2 * context_length + 1))
        visual_context = []
        for t in range(self.context_length, visual_features.shape[-1] - self.context_length):
            visual_context.append(visual_features[:, :, t-self.context_length:t+self.context_length+1].reshape(batch_size, -1))
        visual_context = torch.stack(visual_context, dim=1)

        # Process through MLP
        out = self.mlp(visual_context)
        
        output_features = self.regressor(out)
        output_features = output_features.permute(0, 2, 1)
        output_features = output_features * mask
        
        if self.num_classes > 0:
            output_logits = self.classifier(out)
            output_probs = output_logits 
            output_probs = output_probs.permute(0, 2, 1)
            output_probs = output_probs * mask
            
            return output_probs, output_features
        else:
            return output_features