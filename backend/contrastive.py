"""
Contrastive Learning Operations for Vulkan Backend

GPU-accelerated contrastive learning:
- Contrastive loss
- Contrastive gradient

Uses: contrastive-loss.glsl, contrastive-gradient.glsl
"""

import numpy as np
import struct
from typing import Tuple
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanContrastive:
    """GPU-accelerated contrastive learning operations"""
    
    def __init__(self, core, pipelines, shaders):
        """Initialize with VulkanCore, VulkanPipelines, and shaders dict"""
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    def contrastive_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        temperature: float = 0.07
    ) -> float:
        """
        Compute contrastive loss.
        
        Uses: contrastive-loss.glsl
        
        Args:
            anchor: Anchor embeddings (batch, dim)
            positive: Positive embeddings (batch, dim)
            negative: Negative embeddings (batch, num_negatives, dim)
            temperature: Temperature parameter
        
        Returns:
            Contrastive loss value
        """
        if 'contrastive-loss' in self.shaders:
            # GPU implementation
            batch_size, dim = anchor.shape
            num_negatives = negative.shape[1] if negative.ndim == 3 else negative.shape[0]
            
            anchor_flat = anchor.astype(np.float32).flatten()
            positive_flat = positive.astype(np.float32).flatten()
            negative_flat = negative.astype(np.float32).flatten()
            
            # Output buffers
            losses = np.zeros(batch_size, dtype=np.float32)
            hardest_idx = np.zeros(batch_size, dtype=np.int32)
            pos_dists = np.zeros(batch_size, dtype=np.float32)
            neg_dists = np.zeros(batch_size, dtype=np.float32)
            
            # Create buffers
            buf_anchor, mem_anchor = self.core._create_buffer(anchor_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buf_positive, mem_positive = self.core._create_buffer(positive_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buf_negative, mem_negative = self.core._create_buffer(negative_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buf_losses, mem_losses = self.core._create_buffer(losses.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buf_hardest, mem_hardest = self.core._create_buffer(hardest_idx.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buf_pos_dist, mem_pos_dist = self.core._create_buffer(pos_dists.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buf_neg_dist, mem_neg_dist = self.core._create_buffer(neg_dists.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            
            # Upload
            self.core._upload_buffer(buf_anchor, mem_anchor, anchor_flat)
            self.core._upload_buffer(buf_positive, mem_positive, positive_flat)
            self.core._upload_buffer(buf_negative, mem_negative, negative_flat)
            
            # Get pipeline
            pipeline, layout, desc_layout = self.pipelines.get_or_create_pipeline(
                'contrastive-loss', 7, push_constant_size=16
            )
            desc_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (buf_anchor, anchor_flat.nbytes),
                    (buf_positive, positive_flat.nbytes),
                    (buf_negative, negative_flat.nbytes),
                    (buf_losses, losses.nbytes),
                    (buf_hardest, hardest_idx.nbytes),
                    (buf_pos_dist, pos_dists.nbytes),
                    (buf_neg_dist, neg_dists.nbytes)
                ]
            )
            
            # Dispatch
            margin = 0.2  # Default margin
            push_constants = struct.pack('IIIf', batch_size, dim, num_negatives, margin)
            workgroups = (batch_size + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_constants)
            
            # Download
            losses = self.core._download_buffer(mem_losses, losses.nbytes, dtype=np.float32)
            
            # Cleanup
            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
            vkDestroyBuffer(self.core.device, buf_anchor, None)
            vkDestroyBuffer(self.core.device, buf_positive, None)
            vkDestroyBuffer(self.core.device, buf_negative, None)
            vkDestroyBuffer(self.core.device, buf_losses, None)
            vkDestroyBuffer(self.core.device, buf_hardest, None)
            vkDestroyBuffer(self.core.device, buf_pos_dist, None)
            vkDestroyBuffer(self.core.device, buf_neg_dist, None)
            vkFreeMemory(self.core.device, mem_anchor, None)
            vkFreeMemory(self.core.device, mem_positive, None)
            vkFreeMemory(self.core.device, mem_negative, None)
            vkFreeMemory(self.core.device, mem_losses, None)
            vkFreeMemory(self.core.device, mem_hardest, None)
            vkFreeMemory(self.core.device, mem_pos_dist, None)
            vkFreeMemory(self.core.device, mem_neg_dist, None)
            
            return float(np.mean(losses))
        else:
            # CPU fallback - SimCLR-style contrastive loss
            anchor_norm = anchor / (np.linalg.norm(anchor, axis=1, keepdims=True) + 1e-8)
            positive_norm = positive / (np.linalg.norm(positive, axis=1, keepdims=True) + 1e-8)
            
            # Positive similarity
            pos_sim = np.sum(anchor_norm * positive_norm, axis=1) / temperature
            
            # Negative similarities
            negative_norm = negative / (np.linalg.norm(negative, axis=2, keepdims=True) + 1e-8)
            neg_sims = np.sum(anchor_norm[:, None, :] * negative_norm, axis=2) / temperature
            
            # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            numerator = np.exp(pos_sim)
            denominator = numerator + np.sum(np.exp(neg_sims), axis=1)
            loss = -np.log(numerator / (denominator + 1e-8))
            
            return float(np.mean(loss))
    
    def contrastive_gradient(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        loss: float,
        temperature: float = 0.07
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute contrastive loss gradients.
        
        Uses: contrastive-gradient.glsl
        
        Args:
            anchor: Anchor embeddings (batch, dim)
            positive: Positive embeddings (batch, dim)
            negative: Negative embeddings (batch, num_negatives, dim)
            loss: Contrastive loss value
            temperature: Temperature parameter
        
        Returns:
            (anchor_grad, positive_grad, negative_grad)
        """
        if 'contrastive-gradient' in self.shaders:
            # GPU implementation would go here
            # This is complex and requires capsule gradients, so using CPU fallback for now
            pass
        
        # CPU fallback - compute gradients for contrastive loss
        batch_size, dim = anchor.shape
        num_negatives = negative.shape[1] if negative.ndim == 3 else negative.shape[0]
        
        # Normalize embeddings
        anchor_norm = anchor / (np.linalg.norm(anchor, axis=1, keepdims=True) + 1e-8)
        positive_norm = positive / (np.linalg.norm(positive, axis=1, keepdims=True) + 1e-8)
        negative_norm = negative / (np.linalg.norm(negative, axis=2, keepdims=True) + 1e-8)
        
        # Compute similarities
        pos_sim = np.sum(anchor_norm * positive_norm, axis=1) / temperature
        neg_sims = np.sum(anchor_norm[:, None, :] * negative_norm, axis=2) / temperature
        
        # Compute loss components
        exp_pos = np.exp(pos_sim)
        exp_neg = np.exp(neg_sims)
        exp_neg_sum = np.sum(exp_neg, axis=1, keepdims=True)
        denominator = exp_pos[:, None] + exp_neg_sum
        
        # Gradients w.r.t. anchor
        pos_term = (exp_pos[:, None] / denominator) * (positive_norm / temperature)
        neg_term = np.sum((exp_neg / denominator) * (negative_norm / temperature), axis=1)
        anchor_grad = -pos_term + neg_term
        
        # Gradients w.r.t. positive
        positive_grad = (exp_pos[:, None] / denominator) * (anchor_norm / temperature)
        
        # Gradients w.r.t. negatives
        negative_grad = -(exp_neg / denominator) * (anchor_norm[:, None, :] / temperature)
        
        return anchor_grad, positive_grad, negative_grad
