import cv2
import numpy as np

def reconstruct_image(blended_pyramid):
    """
    从拉普拉斯金字塔重建图像
    """
    image = blended_pyramid[-1]
    for level in range(len(blended_pyramid)-2, -1, -1):
        up = cv2.pyrUp(image, dstsize=(
            blended_pyramid[level].shape[1], 
            blended_pyramid[level].shape[0]
        ))
        image = up + blended_pyramid[level]
    return image