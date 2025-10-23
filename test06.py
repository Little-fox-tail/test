import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt  

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

# 可视化函数
def visualize_alignment(orig_img, aligned_img, save_path):
    fig = plt.figure(figsize=(12,6))
    # 原始图像边缘
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))  # 修正颜色通道
    plt.title('Original Image')
    # 对齐后边缘
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    plt.title('Aligned Image')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
def calculate_dynamic_threshold(image, lower=120, upper=170, max_count=20000):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # 计算灰度直方图
    hist = hist.flatten()  # 将直方图变为一维
    pixel_counts = hist[lower:upper + 1]  # 提取指定范围内的像素计数

    # 寻找局部最小值
    local_minima = []
    for i in range(len(pixel_counts)):
        # 确保索引不会超出边界
        if i >= 10 and i < len(pixel_counts) - 10:
            if pixel_counts[i - 10] > pixel_counts[i] < pixel_counts[i + 10] and pixel_counts[i] < max_count:
                local_minima.append((i, pixel_counts[i]))  # 保存索引和对应的像素计数

    # 如果找到多个局部最小值，选择最中间的一个作为阈值
    if local_minima:
        # 根据像素计数对局部最小值进行排序
        local_minima.sort(key=lambda x: x[1])
        # 选择中间的值
        mid_index = len(local_minima) // 2
        threshold_value = local_minima[mid_index][0] + lower  # 将索引转换回原始的阈值
    else:
        # 如果没有找到局部最小值，选择默认的阈值
        threshold_value = 135

    return threshold_value

# 文件处理
def process_images(base_path, sub_folders_dof, sub_folders_us, sub_folders_group, save_path):
    # 提取所有同名图像进行融合
    image_files = sorted(glob.glob(os.path.join(base_path, sub_folders_dof, sub_folders_us[0], sub_folders_group, '*.bmp')))
    if len(image_files) == 0:
        print(f"No images found in {os.path.join(base_path, sub_folders_dof, sub_folders_us[0], sub_folders_group)}.")
        return

    def check_image_diff(img1, img2):
        diff = cv2.absdiff(img1, img2)
        diff_ratio = np.count_nonzero(diff) / diff.size
        print(f"图像差异率：{diff_ratio*100:.2f}%")
        return diff_ratio

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

        # 调用独立可视化函数
        if len(aligned_images) >= 2:
                # 生成对比图保存路径
                vis_path = os.path.join(save_path, f"alignment_{os.path.splitext(image_name)[0]}.png")
                visualize_alignment(images[1], aligned_images[1], vis_path)  # 对比第2张图
        # 验证输入图像差异
        if len(images) >= 2:
            diff_ratio = check_image_diff(aligned_images[0], aligned_images[1])
            if diff_ratio < 0.05:  # 差异小于5%时报警
                print("警告：输入图像差异过小，可能导致对齐效果不明显")

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

if __name__ == '__main__':
    base_path = 'E:/shuoshi_File/file_241214/43/dingpai'
    base_path_dof = '3.0dof'
    base_path_us = ['40us', '70us', '100us']
    base_path_groups = ['1']  # 定义所有组的列表 , '2', '3', '4', '5'

    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    for group in base_path_groups:
        save_path = f'result/result_{current_file_name}/dingpai/{base_path_dof}/{group}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 拼接完整路径并处理图像
        process_images(base_path, base_path_dof, base_path_us, group, save_path)
        print(f"Processed group: {group}")  # 打印当前处理的组
