"""
grad_cam.py
Công cụ trực quan hóa Grad-CAM cho mô hình PyTorch Transformer.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

# Thêm đường dẫn gốc
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Đăng ký hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Target class output
        # Output shape is (B, 7, H, W). We want the gradients of a specific class.
        target = output[:, class_idx, :, :].sum()
        
        # Backward pass
        target.backward()
        
        # Lấy gradients và activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Nhân weights với activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Phủ Heatmap lên ảnh gốc để dễ quan sát"""
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # img đang ở [0, 1], chuyển về [0, 255]
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
        
    # Nếu ảnh xám (2D) thì chuyển sang 3D
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[0] == 3: # (C, H, W) to (H, W, C)
        img = img.transpose(1, 2, 0)
        
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img

if __name__ == "__main__":
    print("Grad-CAM visualization module ready for PyTorch.")
    print("To use: Initialize GradCAM(model, target_layer) and call cam.generate(image_tensor, target_class_idx)")
