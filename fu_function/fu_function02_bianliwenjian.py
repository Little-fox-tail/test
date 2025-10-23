import cv2
import numpy as np
import sys


# 定义一个函数，用于读取一组图像文件
def readImages():
    # 定义一个包含图像文件路径的列表
    filenames = ["img/duobaoguang2/0007.bmp",
                 "img/duobaoguang2/0009.bmp"]
    # 创建一个空列表，用于存储读取的图像
    images = []
    # 遍历文件名列表，读取每个图像，并将其添加到图像列表中
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    # 返回包含所有读取图像的列表
    return images


# 确保以下代码仅在脚本作为主程序运行时执行
if __name__ == '__main__':
    # 调用函数，读取图像列表
    images = readImages()
    # 创建一个对齐对象，用于对输入图像进行对齐
    alignMTB = cv2.createAlignMTB()
    # 对图像列表进行对齐处理，结果存储在同一个列表中
    alignMTB.process(images, images)

    # 创建一个合并对象，用于曝光融合
    mergeMertens = cv2.createMergeMertens()
    # 使用合并对象处理图像列表，得到曝光融合的结果
    exposureFusion = mergeMertens.process(images)

    # 打印消息，表示开始保存输出图像
    print("Saving output ... exposure-fusion.bmp")
    # 将曝光融合后的结果保存为图像文件，乘以255是为了将数据范围从[0,1]转换为[0,255]
    cv2.imwrite("result2/exposure-fusion4.bmp", exposureFusion * 255)
