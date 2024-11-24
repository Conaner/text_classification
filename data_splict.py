import os
import shutil
import random

def split_data(data_folder, train_ratio=0.7):
    """
    将数据按训练集和测试集比例划分，默认70%训练集，30%测试集
    """
    # 遍历data文件夹中的所有子文件夹
    for cls_folder in os.listdir(data_folder):
        if cls_folder=='train' or cls_folder=='test':
            continue
        cls_path = os.path.join(data_folder, cls_folder)
        
        if os.path.isdir(cls_path):  # 确保是文件夹
            # 获取当前类别下所有的文件列表
            all_files = os.listdir(cls_path)
            random.shuffle(all_files)  # 打乱文件顺序

            # 根据训练集和测试集比例划分
            num_train = int(len(all_files) * train_ratio)
            train_files = all_files[:num_train]
            test_files = all_files[num_train:]

            # 创建训练集和测试集的目标文件夹
            train_folder = os.path.join(data_folder, 'train', 'raw', cls_folder)
            test_folder = os.path.join(data_folder, 'test', 'raw', cls_folder)

            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)

            # 将文件移动到训练集和测试集文件夹
            for train_file in train_files:
                shutil.copy(os.path.join(cls_path, train_file), os.path.join(train_folder, train_file))

            for test_file in test_files:
                shutil.copy(os.path.join(cls_path, test_file), os.path.join(test_folder, test_file))

if __name__ == "__main__":
    # 设置数据集所在的根目录路径
    data_folder = 'data'  
    split_data(data_folder)
    print("数据集划分完成！")
