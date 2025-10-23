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

# 将图像四周宽度为100像素的边缘转成255
def set_edges_to_255(image, edge_width=100):
    height, width = image.shape[:2]
    # 设置顶部和底部边缘
    image[:edge_width, :] = 255
    image[-edge_width:, :] = 255
    # 设置左侧和右侧边缘
    image[:, :edge_width] = 255
    image[:, -edge_width:] = 255
    return image

# 将大于threshold的像素值设置为255
def cap_pixel_values(image, threshold):
    image[image > threshold] = 255
    return image

# 计算动态阈值
def calculate_dynamic_threshold(image, lower=80, upper=170, smoothing_window=5):
    """
    计算动态阈值：
    1. 提取图像灰度直方图中灰度值在lower到upper之间的数据，并对该区间直方图数据进行平滑处理；
    2. 检测平滑后曲线中的局部最小值和局部最大值，并选择满足两侧均有局部最大值的局部最小值作为初步阈值；
    3. 检查灰度图中低于该阈值的像素数占比，如果少于10%（即大部分像素会被置为白），则采用10百分位对应的灰度值作为最低阈值；
    4. 最终返回不低于10百分位的阈值。
    """
    # 将图像转换为灰度（如果为彩色）
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    num_ratio = 10 # 最低阈值比例
    
    # 计算完整直方图，并提取[lower, upper]区间的数据
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    hist_range = hist[lower:upper + 1]

    # 对直方图数据进行平滑处理，采用滑动平均法
    kernel = np.ones(smoothing_window) / smoothing_window
    smooth_hist = np.convolve(hist_range, kernel, mode='same')

    # 检测平滑后的直方图中的局部最小值和局部最大值
    local_minima = []
    local_maxima = []
    for i in range(1, len(smooth_hist) - 1):
        if smooth_hist[i - 1] > smooth_hist[i] < smooth_hist[i + 1]:
            local_minima.append(i + lower)  # 转换回原始灰度值
        if smooth_hist[i - 1] < smooth_hist[i] > smooth_hist[i + 1]:
            local_maxima.append(i + lower)

    # 从局部最小值中选取满足两侧均存在局部最大值的点作为初步阈值
    computed_threshold = None
    for m in local_minima:
        if any(x < m for x in local_maxima) and any(x > m for x in local_maxima):
            computed_threshold = m
            break
    if computed_threshold is None:
        computed_threshold = lower  # 默认值对比10修改

    # 检查低于该阈值的像素数量比例
    total_pixels = gray_image.size
    num_below = np.sum(gray_image <= computed_threshold)
    ratio = num_below / total_pixels

    # 如果低于阈值的像素少于最低比例阈值num_ratio，则调整阈值为灰度图的num_ratio百分位值
    if ratio < (num_ratio*0.01):
        percentile_10 = np.percentile(gray_image, num_ratio)
        # 选取较高的阈值，确保至少有10%的像素不被置白
        computed_threshold = max(computed_threshold, percentile_10)

    return computed_threshold


# 文件处理
def process_images(base_path, sub_folders_dof, sub_folders_us, sub_folders_group, save_path):
    # 提取所有同名图像进行融合
    image_files = sorted(glob.glob(os.path.join(base_path, sub_folders_dof, sub_folders_us[0], sub_folders_group, '*.bmp')))
    if len(image_files) == 0:
        print(f"No images found in {os.path.join(base_path, sub_folders_dof, sub_folders_us[0], sub_folders_group)}.")
        return

    for image_path in image_files:
        image_name = os.path.basename(image_path)
        images = []
        for sub_folder in sub_folders_us:
            image_path = os.path.join(base_path, sub_folders_dof, sub_folder, sub_folders_group, image_name)
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
        # 将图像四周宽度为100像素的边缘转成255
        fusion_result = set_edges_to_255(fusion_result)  
        # save_file = os.path.join(save_path, f"fusion_{os.path.splitext(image_name)[0]}.Bmp")
        # cv2.imwrite(save_file, fusion_result)
        
        # 动态计算threshold并处理像素值
        threshold = calculate_dynamic_threshold(fusion_result)
        fusion_result = cap_pixel_values(fusion_result, threshold)

        # 保存结果
        save_file = os.path.join(save_path, f"fusion_{os.path.splitext(image_name)[0]}_{threshold}.Bmp")
        cv2.imwrite(save_file, fusion_result)

        # 动态计算threshold并处理像素值
        threshold = calculate_dynamic_threshold(fusion_result)
        fusion_result = cap_pixel_values(fusion_result, threshold)

        # 先使用中值滤波去除斑点
        fusion_result = cv2.medianBlur(fusion_result, 3)

        # 如果需要进一步清除斑点，可进行形态学开运算
        kernel = np.ones((5, 5), np.uint8)
        fusion_result = cv2.morphologyEx(fusion_result, cv2.MORPH_OPEN, kernel)

        # 保存结果
        save_file = os.path.join(save_path, f"fusion_{os.path.splitext(image_name)[0]}_{threshold}.Bmp")
        cv2.imwrite(save_file, fusion_result)


# if __name__ == '__main__':
#     base_path = 'E:/shuoshi_File/file_241214/43/dingpai'
#     base_path_dof = '0.0dof'
#     base_path_us = ['40us',  '100us']#'70us',
#     base_path_groups = ['1', '2', '3', '4', '5']  # 定义所有组的列表

#     current_file_name = os.path.splitext(os.path.basename(__file__))[0]
#     for group in base_path_groups:
#         save_path = f'result/result_{current_file_name}/dingpai/{base_path_dof}_10_5/{group}'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         # 拼接完整路径并处理图像
#         process_images(base_path, base_path_dof, base_path_us, group, save_path)
#         print(f"Processed group: {group}")  # 打印当前处理的组
