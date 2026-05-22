"""
NeuroSLAM增强视觉特征提取器
基于HART (Hierarchical Attentive Recurrent Tracking) 和 CORnet的简化实现

参考:
- HART: https://github.com/akosiorek/hart
- CORnet: https://github.com/dicarlolab/CORnet

核心特性:
1. 类脑层次化特征提取 (V1->V2->V4->IT, inspired by CORnet)
2. 注意力机制 (inspired by HART)
3. 时序特征整合 (LSTM, inspired by HART)
4. 轻量级设计，易于集成
"""

import numpy as np
import cv2
from typing import Tuple


class NeuroVisualFeatureExtractor:
    """
    增强的视觉特征提取器
    结合了类脑视觉处理和注意力机制
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (64, 128),
                 feature_dim: int = 256,
                 use_attention: bool = True,
                 use_temporal: bool = False):
        """
        初始化特征提取器
        
        Args:
            input_size: 输入图像尺寸 (height, width)
            feature_dim: 输出特征维度
            use_attention: 是否使用注意力机制
            use_temporal: 是否使用时序特征整合
        """
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        self.use_temporal = use_temporal
        
        # 时序状态 (用于LSTM-like处理)
        self.hidden_state = None
        self.prev_features = None
        
        # 初始化类脑视觉处理kernels (简化版CORnet)
        self._init_brain_kernels()
        
        # 初始化注意力权重
        if use_attention:
            self._init_attention_weights()
    
    def _init_brain_kernels(self):
        """
        初始化类脑视觉处理核心
        模拟V1->V2->V4->IT的层次化处理
        """
        # V1: 简单边缘检测 (Gabor-like filters)
        self.v1_kernels = self._create_gabor_kernels(
            n_orientations=8, 
            n_scales=3
        )
        
        # V2: 复杂特征组合
        self.v2_kernel_size = 5
        
        # V4: 中层特征
        self.v4_kernel_size = 7
        
        # IT: 高层语义特征
        self.it_feature_dim = self.feature_dim
    
    def _create_gabor_kernels(self, n_orientations: int = 8, 
                             n_scales: int = 3) -> list:
        """
        创建Gabor滤波器组 (模拟V1简单细胞)
        
        Args:
            n_orientations: 方向数量
            n_scales: 尺度数量
        
        Returns:
            Gabor核列表
        """
        kernels = []
        for scale_idx in range(n_scales):
            sigma = 2.0 * (scale_idx + 1)
            lambd = sigma * 1.5
            
            for theta in np.linspace(0, np.pi, n_orientations, endpoint=False):
                # 简化的Gabor参数
                kernel_size = int(6 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), 
                    sigma, 
                    theta, 
                    lambd, 
                    0.5, 
                    0, 
                    ktype=cv2.CV_32F
                )
                kernels.append(kernel)
        
        return kernels
    
    def _init_attention_weights(self):
        """
        初始化注意力机制权重 (inspired by HART)
        """
        # 简化的注意力：使用saliency map
        self.attention_kernel = cv2.getGaussianKernel(15, 3.0)
        self.attention_kernel = self.attention_kernel @ self.attention_kernel.T
    
    def _v1_processing(self, img: np.ndarray) -> np.ndarray:
        """
        V1层处理：简单特征检测 (边缘、方向)
        
        Args:
            img: 输入图像 [H, W]
        
        Returns:
            V1特征图 [H, W, N_filters]
        """
        v1_responses = []
        
        for kernel in self.v1_kernels:
            response = cv2.filter2D(img, cv2.CV_32F, kernel)
            v1_responses.append(response)
        
        # Stack所有方向和尺度的响应
        v1_features = np.stack(v1_responses, axis=-1)
        
        # 非线性激活 (ReLU-like)
        v1_features = np.maximum(v1_features, 0)
        
        return v1_features
    
    def _v2_processing(self, v1_features: np.ndarray) -> np.ndarray:
        """
        V2层处理：复杂特征组合

        Args:
            v1_features: V1特征图

        Returns:
            V2特征图
        """
        h, w, c = v1_features.shape
        pool_size = 2
        h_out = h // pool_size
        w_out = w // pool_size

        # 向量化池化：reshape + transpose 避免逐像素循环
        h_trim = h_out * pool_size
        w_trim = w_out * pool_size
        patches = (
            v1_features[:h_trim, :w_trim, :]
            .reshape(h_out, pool_size, w_out, pool_size, c)
            .transpose(0, 2, 1, 3, 4)
            .reshape(h_out * w_out, pool_size * pool_size, c)
        )
        max_pool = np.max(patches, axis=1)
        avg_pool = np.mean(patches, axis=1)
        return np.concatenate([max_pool, avg_pool], axis=1)
    
    def _v4_processing(self, v2_features: np.ndarray) -> np.ndarray:
        """
        V4层处理：中层特征整合
        
        Args:
            v2_features: V2特征
        
        Returns:
            V4特征
        """
        # 进一步降维和特征融合
        # 简化：使用PCA-like的降维
        mean = np.mean(v2_features, axis=0)
        centered = v2_features - mean
        
        # SVD降维
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        
        # 保留主要成分
        n_components = min(128, v2_features.shape[0], v2_features.shape[1])
        v4_features = u[:, :n_components] @ np.diag(s[:n_components])
        
        return v4_features.flatten()
    
    def _it_processing(self, v4_features: np.ndarray) -> np.ndarray:
        """
        IT层处理：高层语义特征 (类似全连接层)
        
        Args:
            v4_features: V4特征
        
        Returns:
            IT特征 (最终输出特征)
        """
        # 降维到目标特征维度
        if len(v4_features) > self.it_feature_dim:
            # 简单的线性投影 + 非线性
            indices = np.linspace(0, len(v4_features)-1, self.it_feature_dim, dtype=int)
            it_features = v4_features[indices]
        else:
            # Padding
            it_features = np.pad(v4_features, 
                                (0, self.it_feature_dim - len(v4_features)), 
                                mode='constant')
        
        # 非线性激活
        it_features = np.tanh(it_features / np.std(it_features + 1e-6))
        
        # L2归一化
        it_features = it_features / (np.linalg.norm(it_features) + 1e-6)
        
        return it_features
    
    def _compute_attention(self, img: np.ndarray) -> np.ndarray:
        """
        计算注意力图 (inspired by HART)
        
        Args:
            img: 输入图像
        
        Returns:
            注意力权重图
        """
        # 简单的saliency detection
        # 1. 强度对比
        img_float = img.astype(np.float32)
        blurred = cv2.GaussianBlur(img_float, (15, 15), 3.0)
        intensity_contrast = np.abs(img_float - blurred)
        
        # 2. 边缘检测
        edges = cv2.Canny(img.astype(np.uint8), 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8))
        
        # 3. 组合
        attention_map = 0.7 * intensity_contrast + 0.3 * edges_dilated
        
        # 归一化
        attention_map = attention_map / (np.max(attention_map) + 1e-6)
        
        return attention_map
    
    def _temporal_integration(self, current_features: np.ndarray) -> np.ndarray:
        """
        时序特征整合 (LSTM-like, inspired by HART)
        
        Args:
            current_features: 当前帧特征
        
        Returns:
            整合后的特征
        """
        if self.prev_features is None:
            # 第一帧
            self.prev_features = current_features
            self.hidden_state = np.zeros_like(current_features)
            return current_features
        
        # 简化的LSTM门控机制
        forget_gate = 0.7  # 固定遗忘门
        input_gate = 0.3   # 固定输入门
        
        # 更新隐藏状态
        self.hidden_state = (forget_gate * self.hidden_state + 
                            input_gate * current_features)
        
        # 输出特征
        integrated_features = np.tanh(self.hidden_state)
        
        self.prev_features = current_features
        
        return integrated_features
    
    def extract_features(self, img: np.ndarray, 
                        apply_attention: bool = None) -> np.ndarray:
        """
        提取图像特征 (主函数)
        
        Args:
            img: 输入图像 [H, W] 或 [H, W, 3]
            apply_attention: 是否应用注意力 (None则使用初始化设置)
        
        Returns:
            特征向量 [feature_dim]
        """
        # 预处理
        if len(img.shape) == 3:
            # 转灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize到标准尺寸
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 注意力机制
        if apply_attention is None:
            apply_attention = self.use_attention
        
        if apply_attention:
            attention_map = self._compute_attention((img * 255).astype(np.uint8))
            img = img * attention_map
        
        # 类脑层次化特征提取
        # V1: 简单特征
        v1_features = self._v1_processing(img)
        
        # V2: 复杂特征
        v2_features = self._v2_processing(v1_features)
        
        # V4: 中层特征
        v4_features = self._v4_processing(v2_features)
        
        # IT: 高层特征
        it_features = self._it_processing(v4_features)
        
        # 时序整合 (可选)
        if self.use_temporal:
            it_features = self._temporal_integration(it_features)
        
        return it_features
    
    def reset_temporal_state(self):
        """重置时序状态"""
        self.hidden_state = None
        self.prev_features = None
    
    def compare_features(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        比较两个特征的相似度
        
        Args:
            feat1, feat2: 特征向量
        
        Returns:
            相似度分数 (0-1, 越大越相似)
        """
        # 余弦相似度
        similarity = np.dot(feat1, feat2) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6
        )
        
        # 转换到0-1范围
        similarity = (similarity + 1) / 2
        
        return similarity


def extract_neuro_features_simple(img: np.ndarray, 
                                  feature_dim: int = 256) -> np.ndarray:
    """
    简化的特征提取函数 (无状态，适合快速调用)
    
    Args:
        img: 输入图像
        feature_dim: 特征维度
    
    Returns:
        特征向量
    """
    extractor = NeuroVisualFeatureExtractor(
        feature_dim=feature_dim,
        use_attention=True,
        use_temporal=False
    )
    return extractor.extract_features(img)


def compare_images_neuro(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    使用神经特征比较两张图像
    
    Args:
        img1, img2: 输入图像
    
    Returns:
        相似度 (0-1)
    """
    extractor = NeuroVisualFeatureExtractor(use_attention=True, use_temporal=False)
    feat1 = extractor.extract_features(img1)
    feat2 = extractor.extract_features(img2)
    return extractor.compare_features(feat1, feat2)


# ============ 与现有NeuroSLAM集成的接口 ============

def neuro_patch_normalization(img: np.ndarray, 
                              patch_size: int = 11) -> np.ndarray:
    """
    增强的Patch Normalization
    替代原有的简单patch normalization
    
    Args:
        img: 输入图像
        patch_size: Patch大小
    
    Returns:
        归一化后的特征图
    """
    # 提取神经特征
    extractor = NeuroVisualFeatureExtractor(
        input_size=(64, 128),
        feature_dim=256,
        use_attention=True,
        use_temporal=False
    )
    features = extractor.extract_features(img)
    
    # 重塑为2D特征图 (方便与原有代码兼容)
    feat_h = 16  # 可调
    feat_w = 16
    feature_map = features[:feat_h*feat_w].reshape(feat_h, feat_w)
    
    # Resize到原图尺寸
    h, w = img.shape[:2]
    feature_map_resized = cv2.resize(feature_map, (w, h))
    
    return feature_map_resized


if __name__ == "__main__":
    """
    测试代码
    """
    print("=== NeuroSLAM增强视觉特征提取器测试 ===\n")
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (120, 240), dtype=np.uint8)
    print(f"测试图像尺寸: {test_img.shape}")
    
    # 测试1: 基础特征提取
    print("\n[测试1] 基础特征提取")
    extractor = NeuroVisualFeatureExtractor(feature_dim=256)
    features = extractor.extract_features(test_img)
    print(f"  特征维度: {features.shape}")
    print(f"  特征范围: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  特征norm: {np.linalg.norm(features):.3f}")
    
    # 测试2: 带注意力的特征提取
    print("\n[测试2] 带注意力机制")
    extractor_attn = NeuroVisualFeatureExtractor(
        feature_dim=256, 
        use_attention=True
    )
    features_attn = extractor_attn.extract_features(test_img)
    print(f"  特征维度: {features_attn.shape}")
    
    # 测试3: 时序特征整合
    print("\n[测试3] 时序特征整合")
    extractor_temp = NeuroVisualFeatureExtractor(
        feature_dim=256,
        use_temporal=True
    )
    for i in range(3):
        feat_t = extractor_temp.extract_features(test_img)
        print(f"  帧{i+1} 特征norm: {np.linalg.norm(feat_t):.3f}")
    
    # 测试4: 图像相似度比较
    print("\n[测试4] 图像相似度比较")
    img1 = test_img.copy()
    img2 = test_img + np.random.randint(-10, 10, test_img.shape).astype(np.uint8)
    similarity = compare_images_neuro(img1, img2)
    print(f"  相似图像相似度: {similarity:.3f}")
    
    img3 = np.random.randint(0, 255, test_img.shape, dtype=np.uint8)
    similarity2 = compare_images_neuro(img1, img3)
    print(f"  随机图像相似度: {similarity2:.3f}")
    
    # 测试5: 与MATLAB集成的接口
    print("\n[测试5] MATLAB集成接口")
    norm_map = neuro_patch_normalization(test_img, patch_size=11)
    print(f"  归一化特征图尺寸: {norm_map.shape}")
    print(f"  特征图范围: [{norm_map.min():.3f}, {norm_map.max():.3f}]")
    
    print("\n=== 测试完成 ===")
