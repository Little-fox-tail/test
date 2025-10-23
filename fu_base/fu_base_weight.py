import cv2
import numpy as np

def compute_weights(images, contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0):
    """
    计算每张图像的权重图
    """
    weights = []
    for img in images:
        img_float = img.astype(np.float32) / 255.0

        # 对比度权重：拉普拉斯滤波后的绝对值
        gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        contrast = np.abs(laplacian) + 1e-6  # 避免除零
        contrast_weight_map = contrast ** contrast_weight

        # 饱和度权重：RGB通道的标准差
        saturation = np.std(img_float, axis=2) + 1e-6
        saturation_weight_map = saturation ** saturation_weight

        # 曝光权重：基于像素亮度与中值（0.5）的距离
        exposure = 0.8  # 理想曝光中值
        intensity = np.mean(img_float, axis=2)
        exposure_weight_map = np.exp(-(intensity - exposure)**2 / (2 * 0.2**2))  # 高斯函数
        exposure_weight_map = exposure_weight_map ** exposure_weight

        # 总权重 = 对比度 * 饱和度 * 曝光度
        weight_map = contrast_weight_map * saturation_weight_map * exposure_weight_map
        weights.append(weight_map)

    # 归一化权重：每像素的所有图像权重和为1
    weights_sum = np.sum(weights, axis=0)
    normalized_weights = [w / (weights_sum + 1e-6) for w in weights]
    return normalized_weights