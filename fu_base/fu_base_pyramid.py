import cv2
import numpy as np

def build_gaussian_pyramid(image, levels):
    """
    生成高斯金字塔
    """
    pyramid = [image]
    for _ in range(levels-1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def build_laplacian_pyramid(image, levels):
    """
    生成拉普拉斯金字塔
    """
    pyramid = []
    for _ in range(levels-1):
        down = cv2.pyrDown(image)
        up = cv2.pyrUp(down, dstsize=(image.shape[1], image.shape[0]))
        pyramid.append(image - up)
        image = down
    pyramid.append(image)  # 最后一层为高斯金字塔顶层
    return pyramid

def blend_pyramids(images, weights, levels=5):
    """
    融合拉普拉斯金字塔和高斯权重金字塔
    """
    # 为每张图像构建拉普拉斯金字塔和高斯权重金字塔
    laplacian_pyramids = [build_laplacian_pyramid(img, levels) for img in images]
    weight_pyramids = [build_gaussian_pyramid(wt, levels) for wt in weights]

    # 融合每一层金字塔
    blended_pyramid = []
    for level in range(levels):
        blended = np.zeros_like(laplacian_pyramids[0][level])
        for i in range(len(images)):
            # 拉普拉斯系数 * 高斯权重
            blended += laplacian_pyramids[i][level] * weight_pyramids[i][level][..., None]
        blended_pyramid.append(blended)

    return blended_pyramid