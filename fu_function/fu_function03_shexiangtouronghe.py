import cv2
from PIL import Image
import numpy as np

# 定义一个函数，用于从摄像头捕获一组图像
def capture_images_from_camera(camera_index, num_images):
    # 创建一个空列表，用于存储捕获的图像
    images = []
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    for i in range(num_images):
        ret, image = cap.read()
        if ret:
            # 将图像添加到列表中
            images.append(image)
        else:
            print(f"Failed to capture image {i+1}")
            break
    # 释放摄像头资源
    cap.release()
    return images

# 定义一个函数，用于进行曝光融合
def exposure_fusion(images):
    # 创建一个对齐对象，用于对输入图像进行对齐
    alignMTB = cv2.createAlignMTB()
    # 对图像列表进行对齐处理，结果存储在同一个列表中
    alignMTB.process(images, images)

    # 创建一个合并对象，用于曝光融合
    mergeMertens = cv2.createMergeMertens()
    # 使用合并对象处理图像列表，得到曝光融合的结果
    exposureFusion = mergeMertens.process(images)

    return exposureFusion

# 确保以下代码仅在脚本作为主程序运行时执行
if __name__ == '__main__':
    # 定义摄像头索引和要捕获的图像数量
    camera_index = 0  # 摄像头序号
    num_images = 100  # 捕获x张不同曝光的图像

    # 调用函数，从摄像头捕获图像列表
    images = capture_images_from_camera(camera_index, num_images)

    # 检查是否成功捕获了所需数量的图像
    if len(images) == num_images:
        # 调用函数，进行曝光融合
        fusion_result = exposure_fusion(images)

        # 确保图像数据在0-255的范围内，并且是uint8类型
        if fusion_result.dtype != np.uint8:
            fusion_result = cv2.normalize(fusion_result, None, 0, 255, cv2.NORM_MINMAX)
            fusion_result = fusion_result.astype(np.uint8)

        # 显示融合后的图像
        cv2.imshow('Exposure Fusion', fusion_result)

        # 保存融合后的图像
        cv2.imwrite(f'exposure_fusion_result{num_images}.jpg', fusion_result)

        # 等待用户按键，然后退出
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough images captured for exposure fusion.")