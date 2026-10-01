"""
architectures.py
Triển khai mô hình SegFormer và Swin Transformer bằng PyTorch + HuggingFace.
"""
import torch
import torch.nn as nn
from transformers import SegformerModel, SwinModel

class SegformerSegmentation(nn.Module):
    def __init__(self, num_classes=7, pretrained="nvidia/mit-b0"):
        super().__init__()
        # Backbone SegFormer (Encoder)
        self.encoder = SegformerModel.from_pretrained(pretrained)
        
        # Vì đầu vào của ta là 3 channels (img, skel, dist), Segformer mặc định nhận 3 channels.
        # Nếu muốn train từ đầu, ta có thể bỏ qua pretrained weights của conv layer đầu tiên, 
        # nhưng tốt nhất cứ giữ nguyên để transfer learning vì ta ghép (img, skel, dist) vào 3 channels.

        # Các dimensions của mit-b0: [32, 64, 160, 256]
        # Xây dựng MLP Decoder đơn giản như trong paper SegFormer
        self.linear_c4 = nn.Conv2d(256, 256, kernel_size=1)
        self.linear_c3 = nn.Conv2d(160, 256, kernel_size=1)
        self.linear_c2 = nn.Conv2d(64, 256, kernel_size=1)
        self.linear_c1 = nn.Conv2d(32, 256, kernel_size=1)
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x shape: (B, 3, H, W)
        outputs = self.encoder(x, output_hidden_states=True)
        features = outputs.hidden_states # List 4 features: 1/4, 1/8, 1/16, 1/32
        
        c1, c2, c3, c4 = features
        
        # Pass qua 1x1 conv
        c4 = self.linear_c4(c4)
        c3 = self.linear_c3(c3)
        c2 = self.linear_c2(c2)
        c1 = self.linear_c1(c1)
        
        # Upsample tất cả về kích thước của c1 (1/4 độ phân giải gốc)
        c4 = nn.functional.interpolate(c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c3 = nn.functional.interpolate(c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c2 = nn.functional.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        
        # Nối lại
        fused = torch.cat([c4, c3, c2, c1], dim=1)
        fused = self.linear_fuse(fused)
        
        # Classifier
        logits = self.classifier(fused)
        
        # Upsample về ảnh gốc
        logits = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

class SwinSegmentation(nn.Module):
    def __init__(self, num_classes=7, pretrained="microsoft/swin-tiny-patch4-window7-224"):
        super().__init__()
        # Backbone Swin (Encoder)
        self.encoder = SwinModel.from_pretrained(pretrained)
        
        # Các dimensions của swin-tiny: [96, 192, 384, 768]
        self.linear_c4 = nn.Conv2d(768, 256, kernel_size=1)
        self.linear_c3 = nn.Conv2d(384, 256, kernel_size=1)
        self.linear_c2 = nn.Conv2d(192, 256, kernel_size=1)
        self.linear_c1 = nn.Conv2d(96, 256, kernel_size=1)
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x shape: (B, 3, H, W)
        outputs = self.encoder(x, output_hidden_states=True)
        features = outputs.hidden_states # Thường trả về 4 features (1/4, 1/8, 1/16, 1/32) cho segmentation backbone
        
        # SwinModel outputs hidden_states as (B, SeqLen, C), we need to reshape them to (B, C, H, W)
        reshaped_features = []
        for i, feat in enumerate(features[1:]): # Bỏ qua embedding input đầu tiên
            B, L, C = feat.shape
            H = W = int(L ** 0.5)
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            reshaped_features.append(feat)
            
        c1, c2, c3, c4 = reshaped_features
        
        c4 = self.linear_c4(c4)
        c3 = self.linear_c3(c3)
        c2 = self.linear_c2(c2)
        c1 = self.linear_c1(c1)
        
        c4 = nn.functional.interpolate(c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c3 = nn.functional.interpolate(c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c2 = nn.functional.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        
        fused = torch.cat([c4, c3, c2, c1], dim=1)
        fused = self.linear_fuse(fused)
        
        logits = self.classifier(fused)
        logits = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

def build_model(model_type="segformer", num_classes=7):
    if model_type == "segformer":
        return SegformerSegmentation(num_classes=num_classes)
    elif model_type == "swin":
        return SwinSegmentation(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
