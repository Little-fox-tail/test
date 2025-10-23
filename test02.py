import os
import fu_function.fu_function11_yuzhi3 as fu

if __name__ == '__main__':
    base_path = 'E:/shuoshi_File/file_241214/43/dingpai'
    base_path_dof = '0.5dof'
    base_path_us = ['40us', '100us']#'70us',
    base_path_groups = ['1']  # 定义所有组的列表, '2', '3', '4', '5'

    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    for group in base_path_groups:
        save_path = f'result/result_{current_file_name}/dingpai/{base_path_dof}/{group}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 拼接完整路径并处理图像
        fu.process_images(base_path, base_path_dof, base_path_us, group, save_path)
        print(f"Processed group: {group}")  # 打印当前处理的组
        print(f"Saved to: {save_path}")  # 打印保存路径