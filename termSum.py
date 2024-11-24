import joblib
import os

def apply_overlapping_kmeans(term_folder_path, num_clusters=8):
    """
    应用重叠聚类
    """
    all_texts = []
    all_term_paths = []

    # 获取所有分词文件路径
    for clsf in os.listdir(term_folder_path):
        cls_folder_path = os.path.join(term_folder_path, clsf)
        if os.path.isdir(cls_folder_path):  # 确保是文件夹
            for text_filename in os.listdir(cls_folder_path):
                term_path = os.path.join(term_folder_path, clsf, text_filename.split('.')[0] + '.pkl')
                # 加载已分词的文本
                with open(term_path, 'rb') as f:
                    terms = joblib.load(f)
                    all_texts.append(" ".join(terms))  # 将分词后的词汇合并成一个字符串
                    all_term_paths.append(term_path)
    with open('sum.txt', 'w', encoding='utf-8') as f:
        # for term_path, text in zip(all_term_paths, all_texts):
        for text in all_texts:
            f.write(f"{text}\n")

if __name__ == '__main__':
    # 训练集分词数据的文件夹路径
    term_file_folder_path = 'data/train/term/'
    apply_overlapping_kmeans(term_file_folder_path)