import fu_function.fu_function13_file_path as fu
import os

if __name__ == '__main__':
    base_path = 'E:/shuoshi_File/file_250312/1'
    base_path_dof = '0.0dof'
    exposure_groups = ['30us', '120us']  # 修正为实际的曝光时间文件夹

    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 创建保存路径
    save_root = f'result/result_{current_file_name}/{base_path_dof}_{exposure_groups[0]}_{exposure_groups[1]}_Blur'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 处理每个曝光组
    fu.process_images(
        base_path=base_path,
        sub_folders_dof=base_path_dof,
        sub_folders_us=exposure_groups,
        save_path=save_root  # 修改保存路径为统一目录
    )
    print(f"All groups processed. Results saved to: {save_root}")  # 打印保存路径