import cv2
import numpy as np

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

# if __name__ == '__main__':
#     base_path = 'E:/shuoshi_File/file_250312/1'
#     base_path_dof = '0.0dof'
#     exposure_groups = ['30us', '120us']  # 修正为实际的曝光时间文件夹

#     current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    
#     # 创建保存路径
#     save_root = f'result/result_{current_file_name}/{base_path_dof}_{exposure_groups[0]}_{exposure_groups[1]}'
#     if not os.path.exists(save_root):
#         os.makedirs(save_root)

#     # 处理每个曝光组
#     process_images(
#         base_path=base_path,
#         sub_folders_dof=base_path_dof,
#         sub_folders_us=exposure_groups,
#         save_path=save_root  # 修改保存路径为统一目录
#     )
#     print(f"All groups processed. Results saved to: {save_root}")  # 打印保存路径