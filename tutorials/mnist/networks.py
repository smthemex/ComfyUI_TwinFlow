import torch
import numpy as np
import torch.nn as nn


class DiffusionUNet(nn.Module):
    def __init__(self, data_dim, conv_hidden_dim, time_embed_dim, num_classes=10, label_embed_dim=32):
        super(DiffusionUNet, self).__init__()
        self.data_dim = data_dim
        self.side_len = int(np.sqrt(data_dim)) # MNIST: 28
        
        # 1. Time Embedding
        # t and tt use separate embedding layers as they are distinct time scalars
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.target_time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # 2. Label Embedding
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        
        # 3. Encoder
        self.enc1 = nn.Sequential(
            (nn.Conv2d(1, conv_hidden_dim, kernel_size=3, padding=1)), 
            nn.ReLU(),
            (nn.Conv2d(conv_hidden_dim, conv_hidden_dim, kernel_size=4, stride=2, padding=1)) 
        )
        self.enc2 = nn.Sequential(
            nn.ReLU(), 
            (nn.Conv2d(conv_hidden_dim, conv_hidden_dim * 2, kernel_size=3, padding=1)),
            nn.ReLU(), 
            (nn.Conv2d(conv_hidden_dim * 2, conv_hidden_dim * 2, kernel_size=4, stride=2, padding=1)) 
        )
        
        # 4. Bottleneck
        self.bottleneck = nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(conv_hidden_dim * 2, conv_hidden_dim * 4, kernel_size=3, padding=1)
        )

        # 5. Condition Projection Layer [Correction]
        # Input = t_emb + tt_emb + label_emb
        total_cond_dim = time_embed_dim * 2 + label_embed_dim
        self.cond_projection = nn.Linear(total_cond_dim, conv_hidden_dim * 4)
        
        # 6. Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(conv_hidden_dim * 4 + conv_hidden_dim * 2, conv_hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(conv_hidden_dim * 2, conv_hidden_dim * 2, kernel_size=3, padding=1)
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(), 
            nn.ConvTranspose2d(conv_hidden_dim * 2 + conv_hidden_dim, conv_hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(conv_hidden_dim, conv_hidden_dim, kernel_size=3, padding=1)
        )
        
        self.output_conv = nn.Conv2d(conv_hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x_flat, t, tt=None, c=None):
        # x_flat: [Batch, 784] -> x: [Batch, 1, 28, 28]
        x = x_flat.view(-1, 1, self.side_len, self.side_len)
        
        # --- Process t ---
        if t.dim() == 1: t = t.unsqueeze(1) # [B, 1]
        t_emb = self.time_embedding(t)      # [B, time_dim]

        # --- Process tt [Correction] ---
        if tt is not None:
            if tt.dim() == 1: tt = tt.unsqueeze(1)
            tt_emb = self.target_time_embedding(tt) # [B, time_dim]
        else:
            # If tt is not provided, fill with zeros (for robustness, though TwinFlow should provide it)
            tt_emb = torch.zeros_like(t_emb)

        # --- Process c (Labels) ---
        if c is not None:
            label_feat = self.label_embedding(c[0]) # [B, label_dim]
        else:
            label_feat = torch.zeros(x.size(0), self.label_embedding.embedding_dim).to(x.device)
        
        # --- Concatenate all conditions ---
        # [B, time_dim + time_dim + label_dim]
        cond_feat = torch.cat([t_emb, tt_emb, label_feat], dim=1)
        
        # --- U-Net Forward ---
        skip1 = self.enc1(x)      
        skip2 = self.enc2(skip1)  
        b = self.bottleneck(skip2) 
        
        # Inject conditions (Bias injection)
        # Project dimensions and reshape to [B, C, 1, 1] for broadcasting addition
        cond_b = self.cond_projection(cond_feat).view(-1, b.shape[1], 1, 1)
        b = b + cond_b 
        
        up1 = self.dec1(torch.cat([b, skip2], dim=1))
        up2 = self.dec2(torch.cat([up1, skip1], dim=1))
        
        output = self.output_conv(up2)
        
        return x_flat - output.view(-1, self.data_dim)


class MLP(nn.Module):

    def __init__(self, in_dim, context_dim, h, out_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim + context_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x, t, tt=None, c=None):
        t = t.flatten().unsqueeze(1)
        tt = tt.flatten().unsqueeze(1)
        if t is not None and tt is None:
            input = torch.cat((x, t), dim=1)
        elif t is not None and tt is not None:
            input = torch.cat((x, t, tt), dim=1)
        else:
            input = x
        return self.network(input)