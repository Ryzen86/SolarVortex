import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FineDetailBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Two standard 3x3 convs for preserving high-frequency features (cracks, edges)
        self.net = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
            ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class ContextBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Dilated convolutions to expand receptive field without losing resolution (inactive regions, halos)
        self.net = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3, padding=2, dilation=2),
            ConvBNReLU(channels, channels, kernel_size=3, padding=4, dilation=4)
        )

    def forward(self, x):
        return self.net(x)

class ClassAwareFusionGate(nn.Module):
    def __init__(self, decoder_channels, reduction=8):
        super().__init__()
        # SE-style gate based on decoder features
        mid_channels = max(16, decoder_channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(decoder_channels, mid_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, 2, 1, bias=False) # 2 outputs: fine and context weights

    def forward(self, decoder_features):
        x = self.gap(decoder_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) # Shape: [B, 2, 1, 1]
        weights = F.softmax(x, dim=1)
        return weights

class DualPathCAFusionSkip(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_dual_path=True, use_dynamic_weights=True):
        super().__init__()
        self.use_dual_path = use_dual_path
        self.use_dynamic_weights = use_dynamic_weights
        
        self.conv_skip = nn.Conv2d(encoder_channels, decoder_channels, 1, bias=False)
        
        if self.use_dual_path:
            self.fine_branch = FineDetailBranch(decoder_channels)
            self.context_branch = ContextBranch(decoder_channels)
        
        if self.use_dynamic_weights:
            self.fusion_gate = ClassAwareFusionGate(decoder_channels)
        elif self.use_dual_path:
            # Fixed 50/50 fusion if dual path but no dynamic weights
            self.register_buffer('fixed_weights', torch.tensor([0.5, 0.5]).view(1, 2, 1, 1))

    def forward(self, encoder_features, decoder_features):
        # 1. Project encoder features to decoder channel dimension
        skip_features = self.conv_skip(encoder_features)
        
        if not self.use_dual_path:
            # Baseline / No dual path: just pass the skip features through
            return skip_features
            
        # 2. Extract fine and context representations
        fine_feat = self.fine_branch(skip_features)
        context_feat = self.context_branch(skip_features)
        
        # 3. Compute fusion weights
        if self.use_dynamic_weights:
            weights = self.fusion_gate(decoder_features) # Shape: [B, 2, 1, 1]
        else:
            weights = self.fixed_weights # Shape: [1, 2, 1, 1]
            
        # 4. Fuse
        w_fine = weights[:, 0:1, :, :]
        w_context = weights[:, 1:2, :, :]
        fused_features = w_fine * fine_feat + w_context * context_feat
        
        return fused_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_fusion=False, use_dual_path=True, use_dynamic_weights=True):
        super().__init__()
        self.use_fusion = use_fusion
        
        if self.use_fusion:
            self.fusion_skip = DualPathCAFusionSkip(skip_channels, in_channels, use_dual_path, use_dynamic_weights)
            concat_channels = in_channels * 2
        else:
            self.conv_skip = nn.Conv2d(skip_channels, in_channels, 1, bias=False) if skip_channels > 0 else None
            concat_channels = in_channels * 2 if skip_channels > 0 else in_channels
            
        self.conv1 = ConvBNReLU(concat_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        # Upsample decoder features
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if skip is not None:
            if self.use_fusion:
                # Apply CA-Fusion to skip features
                processed_skip = self.fusion_skip(skip, x)
                x = torch.cat([processed_skip, x], dim=1)
            else:
                # Standard skip
                processed_skip = self.conv_skip(skip)
                x = torch.cat([processed_skip, x], dim=1)
                
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class CAFUNet(nn.Module):
    def __init__(self, 
                 encoder_name='resnet34', 
                 in_channels=1, 
                 classes=5, 
                 decoder_channels=(256, 128, 64, 32, 16),
                 use_dual_path=True,
                 use_dynamic_weights=True):
        super().__init__()
        
        # 1. Encoder (from timm)
        self.encoder = timm.create_model(
            encoder_name, 
            pretrained=True, 
            in_chans=in_channels, 
            features_only=True
        )
        
        # Get encoder feature channels (dynamically checking)
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 256, 256)
            enc_features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in enc_features] # e.g. [64, 64, 128, 256, 512] for resnet34
            
        encoder_channels = encoder_channels[::-1] # Reverse to match decoder order [512, 256, 128, 64, 64]
        
        # 2. Decoder
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[0] # Deepest encoder feature (e.g. 512)
        
        for i in range(len(decoder_channels)):
            out_ch = decoder_channels[i]
            skip_ch = encoder_channels[i+1] if i+1 < len(encoder_channels) else 0
            
            # Apply CA-Fusion only at the 2 deepest stages (i=0, i=1)
            use_fusion = (i < 2) and (skip_ch > 0)
            
            self.blocks.append(
                DecoderBlock(
                    in_channels=in_ch, 
                    skip_channels=skip_ch, 
                    out_channels=out_ch,
                    use_fusion=use_fusion,
                    use_dual_path=use_dual_path,
                    use_dynamic_weights=use_dynamic_weights
                )
            )
            in_ch = out_ch
            
        # 3. Segmentation Head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        
        # features list for resnet34: [stride 2, stride 4, stride 8, stride 16, stride 32]
        # features: [feat1, feat2, feat3, feat4, feat5]
        
        # Reverse features to match decoder order: [feat5, feat4, feat3, feat2, feat1]
        features = features[::-1]
        
        x = features[0] # Deepest feature
        skips = features[1:]
        
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
            
        # Final upsampling if needed to reach input resolution (stride 2 of the first encoder block)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.segmentation_head(x)
        
        return out
