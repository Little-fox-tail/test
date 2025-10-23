import cv2
import numpy as np
import os
import glob

# 使用光流法对齐图像
def align_images_with_optical_flow(images):
    aligned_images = []
    reference_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    aligned_images.append(images[0])  # 添加参考图像

    for i in range(1, len(images)):
        gray_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(reference_image, gray_image, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        h, w = flow.shape[:2]
        flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
        flow_map_x = (flow_map_x + flow[:, :, 0]).astype(np.float32)
        flow_map_y = (flow_map_y + flow[:, :, 1]).astype(np.float32)

        aligned_image = cv2.remap(images[i], flow_map_x, flow_map_y, interpolation=cv2.INTER_LINEAR)
        aligned_images.append(aligned_image)

    return aligned_images

# 曝光融合
def exposure_fusion(images):
    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)
    return exposureFusion

# 裁剪黑边函数
def crop_black_borders(image, border_size=10):
    return image[border_size:-border_size, border_size:-border_size]

# 将大于threshold的像素值设置为255
def cap_pixel_values(image, threshold):
    image[image > threshold] = 255
    return image

# 计算动态阈值
def calculate_dynamic_threshold(image, lower=130, upper=160):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # 计算灰度直方图
    hist = hist.flatten()  # 将直方图变为一维
    pixel_counts = hist[lower:upper + 1]  # 提取100到150范围内的像素计数
    threshold_value = np.argmin(pixel_counts) + lower  # 找到占比最小的灰度值
    return threshold_value

# 文件处理
def process_images(base_path, sub_folders_dof, sub_folders_Hz, sub_folders_us, sub_folders_group, save_path):
    # 提取所有同名图像进行融合
    image_files = sorted(glob.glob(os.path.join(base_path, sub_folders_dof, sub_folders_Hz, sub_folders_us[0], sub_folders_group, '*.bmp')))
    if len(image_files) == 0:
        print(f"No images found in {os.path.join(base_path, sub_folders_dof, sub_folders_Hz, sub_folders_us[0], sub_folders_group)}.")
        return

    for image_path in image_files:
        image_name = os.path.basename(image_path)
        images = []
        for sub_folder in sub_folders_us:
            image_path = os.path.join(base_path, sub_folders_dof, sub_folders_Hz, sub_folder, sub_folders_group, image_name)
            if os.path.exists(image_path):
                images.append(cv2.imread(image_path))
            else:
                print(f"Image not found: {image_path}")
                continue

        if len(images) != len(sub_folders_us):
            print(f"Skipping {image_name}: Not all exposures are available.")
            continue

        # 对齐并融合图像
        aligned_images = align_images_with_optical_flow(images)
        fusion_result = exposure_fusion(aligned_images)

        # 转换并裁剪
        if fusion_result.dtype != np.uint8:
            fusion_result = cv2.normalize(fusion_result, None, 0, 255, cv2.NORM_MINMAX)
            fusion_result = fusion_result.astype(np.uint8)

        fusion_result = crop_black_borders(fusion_result)

        # 动态计算threshold并处理像素值
        threshold = calculate_dynamic_threshold(fusion_result)
        # print(f"Dynamic threshold for {image_name}: {threshold}")
        fusion_result = cap_pixel_values(fusion_result, threshold)

        # 保存结果
        save_file = os.path.join(save_path, f"fusion_{os.path.splitext(image_name)[0]}_{threshold}.Bmp")
        cv2.imwrite(save_file, fusion_result)

if __name__ == '__main__':
    base_path = 'E:/shuoshi_File/file_241214/43/feipai'
    base_path_dof = '0.5dof'
    base_path_Hz = '0Hz'
    base_path_us = ['40us', '70us', '100us']
    base_path_group = '1'

    save_path = f'result/result_main08/feipai/{base_path_dof}/{base_path_Hz}/{base_path_group}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 拼接完整路径并处理图像
    process_images(base_path, base_path_dof, base_path_Hz, base_path_us, base_path_group, save_path)
