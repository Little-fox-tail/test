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

        # 生成网格坐标，并根据光流生成变换后的坐标映射
        h, w = flow.shape[:2]
        flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
        flow_map_x = (flow_map_x + flow[:, :, 0]).astype(np.float32)
        flow_map_y = (flow_map_y + flow[:, :, 1]).astype(np.float32)

        # 使用光流变换将当前帧对齐到参考帧
        aligned_image = cv2.remap(images[i], flow_map_x, flow_map_y, interpolation=cv2.INTER_LINEAR)
        aligned_images.append(aligned_image)

    return aligned_images

# 定义一个函数，用于曝光融合
def exposure_fusion(images):
    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)
    return exposureFusion

# 裁剪黑边函数
def crop_black_borders(image, border_size=10):
    return image[border_size:-border_size, border_size:-border_size]

# 文件处理
def process_folder(folder_path, save_path):
    # 获取文件夹内所有图像的路径
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.bmp')))
    images = [cv2.imread(path) for path in image_paths]

    if len(images) > 0:
        # 使用光流法对齐图像
        aligned_images = align_images_with_optical_flow(images)

        # 进行曝光融合
        fusion_result = exposure_fusion(aligned_images)

        # 将图像数据转换为 uint8 类型
        if fusion_result.dtype != np.uint8:
            fusion_result = cv2.normalize(fusion_result, None, 0, 255, cv2.NORM_MINMAX)
            fusion_result = fusion_result.astype(np.uint8)

        # 裁剪黑边
        fusion_result = crop_black_borders(fusion_result)

        # 保存融合后的图像
        folder_name = os.path.basename(folder_path)
        cv2.imwrite(f'{save_path}\\exposure_fusion_result_{folder_name}.bmp', fusion_result)

if __name__ == '__main__':
    save_path = 'result/result_main06/duobaoguang03'
    base_path = 'img/duobaoguang03'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 遍历01、02和03文件夹
    for folder in ['01', '02', '03']:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue  # 如果文件夹不存在，则跳过

        process_folder(folder_path, save_path)